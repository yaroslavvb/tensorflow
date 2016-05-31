"""Implementation of ITensor for the immediate API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

class ITensor(object):
  """This class is a Python wrapper over underlying persistent tensor object.
  It tries to provide some compatibility with existing Tensor objects while
  providing Python numeric interface that automatically run operations in
  associated TensorFlow runtime."""

  def __init__(self, env, handle):
    """Initialize ITensor.

    Args:
      env: immediate.Env object
      handle: session_ops.TensorHandle object
    """
    self.env = env
    self.handle = handle

  @property
  def tf_handle(self):
    """Give string handle representing this itensor in TF runtime.
    This string handle is suitable for feeding get_session_tensor op."""
    return self.handle.handle

  @property
  def dtype(self):
    """Tensorflow dtype of given tensor."""
    return self.handle._dtype

  def as_numpy(self):
    """Convert current ITensor into numpy array."""

    return self.env.handle_to_numpy(self.handle)

  # Some user-generated ops call shape inference functions
  # For compatibility with those functions, make this Tensor act like an op
  # with 3 unknown-shaped inputs.
  @property
  def op(self):
    """Method for compatibility with Tensor."""
    node_def = graph_pb2.NodeDef()
    node_def.name = "immediate-dummy-node"
    node_def.input.extend(["dummy1", "dummy2", "dummy3"])

    dummy_input1 = array_ops.placeholder(self.dtype)
    dummy_input2 = array_ops.placeholder(self.dtype)
    dummy_input3 = array_ops.placeholder(self.dtype)
    dummy_op = tf_ops.Operation(node_def, tf_ops.Graph(), inputs=[dummy_input1,
                                                                  dummy_input2,
                                                                  dummy_input3])

    return dummy_op


  def eval(self):
    """Method for compatiblity with Tensor."""
    return self.as_numpy()

  # TODO(yaroslavvb): replace this with TensorShape(None) to avoid unexpected
  # run calls once all static shape inference functions support Unknown shape
  def get_shape(self):
    """Method for compatibility with Tensor."""
    shape_tensor = self.env.tf.shape(self)
    return tensor_shape.TensorShape(tuple(shape_tensor.as_numpy()))

  # Immediate tensors don't have static shape, but keep this method
  # for compatibility with array_ops.py
  def set_shape(self, _unused_shape):
    """Method for compatiblity with Tensor."""
    pass

  def __repr__(self):
    return "iTensor(%s, dtype=%s)" % (self.as_numpy(), self.dtype)

  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types

  def __add__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.add(self, other)

  def __radd__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.add(other, self)

  def __neg__(self):
    return self.env.tf.neg(self)


  def __sub__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.sub(self, other)

  def __rsub__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.sub(other, self)

  def __mul__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.mul(self, other)

  def __rmul__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.mul(other, self)

  def __bool__(self):
    return bool(self.as_numpy())

  def __nonzero__(self):
    return self.__bool__()

  def __lt__(self, other):
    return self.env.tf.less(self, other)

  def __le__(self, other):
    return self.env.tf.less_equal(self, other)

  def __eq__(self, other):
    return self.env.tf.equal(self, other)

  def __ne__(self, other):
    return self.env.tf.not_equal(self, other)

  def __gt__(self, other):
    return self.env.tf.greater(self, other)

  def __ge__(self, other):
    return self.env.tf.greater_equal(self, other)
