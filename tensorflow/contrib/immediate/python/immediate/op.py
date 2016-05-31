from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Op", "OpFactory", "OpWrapper", "PythonOpWrapper"]

from .tensor import Tensor
from .tensor import _ENABLE_DEBUG_LOGGING

from tensorflow.python.framework import ops as tf_ops

import sys

import wrapping_util

# Implementation of Immediate Op
class Op(object):

  def __init__(self, env, input_holders, output_handle, key,
               converted_tensors={}):
    """Initialize Op.

    Args:
      converted_tensors: dictionary of argument position -> converted immedate
        Tensor (only for argument positions where this conversion was made)
    """

    self.env = env
    self.input_holders = input_holders
    self.output_handle = output_handle
    self.key = key
    self.converted_tensors = converted_tensors

  def __call__(self, *args):
    if not len(args) == len(self.input_holders):
      raise ValueError("Too many arguments provided (%d), %s can only accept "
                       "%d" % (len(args), self.__str__(),
                               len(self.input_holders)))

    feed_dict = {}
    for (i, (itensor, holder)) in enumerate(zip(args, self.input_holders)):
      if i in self.converted_tensors:
        itensor = self.converted_tensors[i]
      if not isinstance(itensor, Tensor):
        raise ValueError("All positional arguments must be immediate "
                         "Tensors, instead we see "+str(itensor))
      feed_dict[holder] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    if isinstance(tensor_handle, list):
      return [Tensor(self.env, t) for t in tensor_handle]
    else:
      return Tensor(self.env, tensor_handle)

  def __str__(self):
    return "Op%s" % (str(self.key))

  def __repr__(self):
    return self.__str__()

# Implementation of Immediate Op with keyword call arguments
class KeywordOp(object):

  def __init__(self, env, input_holders, output_handle):
    """Initialize Op.

    Args:
      converted_tensors: dictionary of argument position -> converted immedate
        Tensor (only for argument positions where this conversion was made)
    """

    self.env = env   # used to issue .run calls
    self.input_holders = input_holders
    self.output_handle = output_handle   # tensorflow.TensorHandle

  def __call__(self, **kwargs):

    feed_dict = {}
    for (argname, itensor) in kwargs.items():
      if isinstance(itensor, list):
        holder_list = self.input_holders[argname]
        tensor_list = itensor
        for holder,subtensor in zip(holder_list, tensor_list):
          feed_dict[holder] = subtensor.tf_handle
      else:
        feed_dict[self.input_holders[argname]] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    if isinstance(tensor_handle, list):
      return [Tensor(self.env, t) for t in tensor_handle]
    else:
      return Tensor(self.env, tensor_handle)

  def __str__(self):
    return "Op%s" % (str(self.key))

  def __repr__(self):
    return self.__str__()


def _fixup_args(symbol_name, *args, **kwargs):
  #  print("calling fixup_args for %s, %s, %s" % (symbol_name, args, kwargs))
  
  if symbol_name == 'gen_math_ops._sum':
    # handle gen_math_ops._sum(tensor,tensor,keep_dims)
    if len(args)==3:
      kwargs['keep_dims'] = args[2]
      return args[:-1], kwargs

  # handle gen_random_ops._random_uniform(shape, dtype)
  # move dtype to keyword argument
  if symbol_name == 'gen_random_ops._random_uniform':
    if len(args)==2:
      kwargs['dtype'] = args[1]
      return args[:-1], kwargs
    
  # handle _split(num_split=3, split_dim=1, value=itensor)
  if symbol_name == "gen_array_ops._split":
    if 'value' in kwargs:
      new_args = args+(kwargs['value'],)
      del kwargs['value']
      return new_args, kwargs

  return args, kwargs

def _unfixup_args(symbol_name, *args, **kwargs):
  # for a native function that expects Tensor arguments to be positional
  # move those kwargs back to be positional arguments
  new_args = list(args)

  # handle following usage: gen_array_ops._split(value, split_dim=..)
  # move "value" back into keyword argument
  if symbol_name == 'gen_array_ops._split':
    if 'split_dim' in kwargs and len(args)==1:
      kwargs['value'] = args[0]
      return (), kwargs

  return args, kwargs

# Implementating of OpFactory with graph caching
# TODO(yaroslavvb): get rid of "symbol_name" argument
# TODO(yaroslavvb): store graph separately?
class OpFactory(object):
  """ This object contains the logic needed to wrap TensorFlow native
  operations into versions that operate on persistent tensor handles. It has
  the following state:
  1. Tensorflow Graph. It grows the graph when necessary, but returns existing
      nodes from the graph when possible
  2. immedate.Env -- it defers to Env to convert arguments to immediate tensors
      when user calls it with Python types instead of immediate.Tensors

  The interface is meant to mirror original tf function calls:
  ie, to get functionality of tf.add(1,2), one would call
      result_op = op_factory(tf.add, 1, 2)

  result_op is an immediate.Op object op that can be given immediate.Tensors
  to produce immediate.Tensor result
      result = result_op(immediate.Tensor(1), immediate.Tensor(2))

  Main functionality is in __call__ which goes through following:
  1. Figure out what OpDef corresponds to given op/argument combination
  2. Check if this OpDef has already been created in the graph
  3. If not, call the op to construct the OpDef and connect it to
      tensor handle version ops to get functionality that works on tensor
      handles. Wrap the result in immediate.Op object. Add it to cache.
  4. Retrieve corresponding immediate.Op from cache and return it.
 """

  def __init__(self, env):
    self.env = env
    self.cache = {}

  def __call__(self, symbol_name, symbol, *args, **kwargs):
    if _ENABLE_DEBUG_LOGGING:
      print("OpFactory __call__: %s(%s, %s)" % (symbol_name, args, kwargs))
    
    # create the key to see if the op has been created before
    key = [symbol_name]

    # converted_args stores args converted to Tensors, ie, Python list [1]
    # becomes immediate.Tensor([1])), immediate.Tensor objects are unchanged
    assert len(args) == 0
    converted_args = []  
    converted_tensors = {}
    for i,itensor in enumerate(args):
      if not isinstance(itensor, Tensor):
        try:
          itensor = self.env.numpy_to_tensor(itensor)
          converted_tensors[i] = itensor
        except ValueError as e:
          raise ValueError("All positional arguments must be immediate "
                           "Tensors, or convertible to immediate Tensors "
                           "instead we see %s (numpy error: %s)" % (
                            str(itensor), sys.exc_info()[0]))
      converted_args.append(itensor)
      key.append(itensor.dtype)

    # TODO(yaroslavvb): use signature binding to fill out default kwargs
    # otherwise may get cache miss
    for kwarg_key in sorted(kwargs.keys()):
      key.append(kwarg_key)
      key.append(str(kwargs[kwarg_key]))
    
    # convert to tuple to make it hashable
    key = tuple(key)

    if key in self.cache:
      return self.cache[key]
    
    # create the op
    with self.env.g.as_default():
      # convert args to TensorHandles
      # connect up things
      input_tensors = []
      input_holders = []
      for itensor in converted_args:
        holder, tensor = self.env.get_session_tensor(itensor.dtype)
        input_holders.append(holder)
        input_tensors.append(tensor)

      # extra check, make sure the user didn't use TF-style
      # kwargs approach to specify inputs
      for name,val in kwargs.items():
        if isinstance(val, Tensor):
          raise ValueError("Found Tensor in a keyword argument, use "
                           "positional arguments instead.")


     #      input_tensors, kwargs = _unfixup_args(symbol_name, *input_tensors, **kwargs)

  #      if _ENABLE_DEBUG_LOGGING:
  #        print("OpFactory unfixup: %s(%s, %s)" % (symbol_name, args, kwargs))
      if _ENABLE_DEBUG_LOGGING:
        print("OpFactory redirect: %s(%s, %s)" % (symbol_name, args, kwargs))
      output = symbol(*input_tensors, **kwargs)

      # TODO(yaroslavvb): allow for multiple return values like tf.split
      if isinstance(output, list):
        #  raise ValueError("Only support TF ops that return a single Tensor.")
        output_handle = [self.env.get_session_handle(o) for o in output]
      else:
        output_handle = self.env.get_session_handle(output)

    op = Op(self.env, input_holders, output_handle, key, converted_tensors)
    self.cache[key] = op

    return op


  def _create_key(self, opname, *args, **kwargs):
    return opname

# TODO(yaroslavvb): remove namespace from Op init
class OpWrapper(object):
  """A callable object that mirrors TF generated wrapper, but with immediate
  execution semantics."""

  def __init__(self, namespace, env, symbol_name, symbol):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name  # name of function, ie "tf.nn.relu"
    self.symbol = symbol  # function object
  
  def __call__(self, *args, **kwargs):
    op = self.env.op_factory(self.symbol_name, self.symbol, *args, **kwargs)
    args, kwargs = _fixup_args(self.symbol_name, *args, **kwargs)
    return op(*args)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return "OpWrapper(%s, %s, %s, %s)"%(self.namespace, self.env,
                                        self.symbol_name, self.symbol)

# TODO(yaroslavvb): copy __doc__ and __file__ from the original fun object
class OpWrapper2(object):
  """A callable object that mirrors TF generated wrapper, but with immediate
  execution semantics."""

  def __init__(self, env, symbol):
    self.env = env
    self.symbol = symbol  # function object
  
  def __call__(self, *args, **kwargs):
    op = self.env.op_factory(self.symbol.__name__, self.symbol, *args, **kwargs)
    #    args, kwargs = _fixup_args(self.symbol.__name__, *args, **kwargs)
    return op(*args)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return "OpWrapper2(%s, %s)"%(self.env, self.symbol)


op_input_argnames, op_input_argtypes = wrapping_util.get_op_input_argnames_argtypes()

class OpDefLibraryWrapper(object):
  def __init__(self, env, original_op_def_library):
    self.env = env
    self.original_op_def_library = original_op_def_library

  def apply_op(self, op_type_name, name=None, **keywords):

    if _ENABLE_DEBUG_LOGGING:
      print("OpFactory __call__: %s(%s, %s)" % (symbol_name, args, kwargs))

    # converted_args stores args converted to Tensors, ie, Python list [1]
    # becomes immediate.Tensor([1])), immediate.Tensor objects are unchanged
    itensor_args = {} 
    converted_tensors = {}
    input_names = op_input_argnames[op_type_name]
    input_types = op_input_argtypes[op_type_name]

    old_tensor_inputs = {}
    key = [op_type_name]

    def try_convert_to_itensor(itensor):
      if isinstance(itensor, Tensor):
        return itensor

      if isinstance(itensor, tf_ops.Tensor):
        raise ValueError("Trying to feed a non-immediate Tensor %s to immediate op %s" %
                         (itensor, op_type_name))
      try:
        return self.env.numpy_to_tensor(itensor)
      except ValueError as e:
        raise ValueError("Couldn't convert input argument %s=%s to immediate "
                         "tensor (%s)" % (input_name, itensor,
                                          sys.exc_info()))
        

    for input_name in input_names:
      itensor = keywords[input_name]
      if input_types[input_name] == "list":
        for i in range(len(itensor)):
          itensor[i] = try_convert_to_itensor(itensor[i])
      else:
        itensor = try_convert_to_itensor(itensor)
          
      itensor_args[input_name] = itensor
      # TODO(yaroslavvb): do something about caching with attribute lists
      #      key.append(itensor.dtype)

    with self.env.g.as_default():
      input_holders = {}
      for input_name in input_names:
        if isinstance(itensor_args[input_name], list):
          holder_list = []
          tensor_list = []
          for subtensor in itensor_args[input_name]:
            holder, tensor = self.env.get_session_tensor(subtensor.dtype)
            holder_list.append(holder)
            tensor_list.append(tensor)
          keywords[input_name] = tensor_list
          input_holders[input_name] = holder_list
        else:
          holder, tensor = self.env.get_session_tensor(itensor_args[input_name].dtype)
          #        print("Created %s for %s" %(tensor, input_name))
          input_holders[input_name] = holder
          # replace previous inputs with tensorflow.Tensor args
          keywords[input_name] = tensor

      output = self.original_op_def_library.apply_op(op_type_name,
                                                     **keywords)

      if isinstance(output, list) or isinstance(output, tuple):
        output_handle = [self.env.get_session_handle(o) for o in output]
      elif isinstance(output, tf_ops.Tensor):
        output_handle = self.env.get_session_handle(output)
      else:
        raise ValueError("Op %s gave output (%s) of unexpected type (%s)"
                         % (op_type_name, output, type(output)))

    op = KeywordOp(self.env, input_holders, output_handle)
    return op(**itensor_args)
    #    self.cache[key] = op


class PythonOpWrapper(object):
  """A callable object that mirrors Python tensorflow function."""

  def __init__(self, namespace, env, symbol_name, symbol, global_sub):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name
    self.symbol = symbol
    self.global_sub = global_sub
    
    for global_name in global_sub:
      symbol.__globals__[global_name] = global_sub[global_name]

  def __call__(self, *args, **kwargs):
    if _ENABLE_DEBUG_LOGGING:
      print("%s, %s, %s" % (self, args, kwargs))

    return self.symbol(*args, **kwargs)

  def __str__(self):
    return "PythonOpWrapper(%s, %s, %s, %s)" % (self.namespace, self.env,
                                                self.symbol_name, self.symbol)

  def __repr__(self):
    return "PythonOpWrapper(%s, %s, %s, %s, %s)" % (self.namespace, self.env,
                                                self.symbol_name, self.symbol,
                                                self.global_sub)


# TODO(yaroslavvb): factor out into SimpleWrapper

class ConstantOpWrapper(object):
  """A callable object that mirrors tf.constant."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol
    
  def __call__(self, *args, **kwargs):
    return self.env.constant(*args, **kwargs)

class ConvertToTensorWrapper(object):
  """A callable object that mirrors tf.convert_to_tensor in Immediate
  environment."""

#  def __init__(self, namespace, env, symbol_name):
  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol
    
  def __call__(self, value, dtype=None, name=None, as_ref=False):
    if isinstance(value, Tensor):
      return value
    return self.env.numpy_to_tensor(value, dtype)

class ConcatOpWrapper(object):
  """A callable object that mirrors gen_array_ops._concat"""

  def __init__(self, namespace, env, symbol_name):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name
    
    
    
#  def __call__(self, concat_dim=concat_dim, values=values, name=name):
#    return self.env.numpy_to_tensor(value, dtype)
