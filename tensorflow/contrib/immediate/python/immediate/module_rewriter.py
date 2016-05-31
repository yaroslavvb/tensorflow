"""Contains ModuleRewriter class + helper methods useful for writing custom
symbol_rewriter functions to be used with ModuleRewriter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import sys
import types

__all__ = ["ModuleRewriter"]

def get_symbol_file(symbol):
  """Returns filename of symbol definition, empty string if not available."""

  if hasattr(symbol, "__file__"):
    return symbol.__file__
  else:
    try:
      symbol_module = sys.modules[symbol.__module__]
      return symbol_module.__file__
    except (AttributeError, KeyError):
      return ""


def get_symbol_name(symbol):
  """Returns __name__ attribute or empty string if not available."""
  if hasattr(symbol, "__name__"):
    return symbol.__name__
  else:
    return ""


def copy_function(old_func, updated_module):
  """Copies a function, updating it to point to given module."""

  new_func = types.FunctionType(old_func.__code__, updated_module.__dict__,
                                name=old_func.__name__,
                                argdefs=old_func.__defaults__,
                                closure=old_func.__closure__)
  new_func.__dict__.update(old_func.__dict__)
  new_func.__module__ = updated_module.__name__
  return new_func


class ModuleRewriter(object):
  """Object that controls rewriting of module."""

  def __init__(self, symbol_rewriter, module_prefix="newmodule."):
    """Initialize ModuleRewriter.

    Args:
      symbol_rewriter: callable object that implements symbol rewriting. It
          should accepts a symbol (ie, a function) and return new symbol that
          acts as a replacement, or None to keep original symbol unchanged.
          The name of the symbol should remain unchanged because it's used
          to resolve references from other modules.
      module_prefix: a string that is prefixed to __name__ and __file__
          attributes of copied modules. Because we add new modules to
          sys.modules, this string must be non-empty.
    """

    assert module_prefix, "Module prefix must be non-empty"

    self.symbol_rewriter = symbol_rewriter
    self.module_prefix = module_prefix

    self._done_modules = {}  # dict of old_module->new_module
    self._module_stack = []  # stack of modules to detect cycles


  def __call__(self, original_module):
    return self._rewrite_module(original_module)

  def _rewrite_module(self, original_module):
    """Apply symbol_rewriter to given module and its dependencies recursively
    and return the result. Copies of objects are made as necessary and original
    module remains unchanged.

    Args:
      original_module: module to rewrite.

    Returns:
      Copy of module hierarchy with rewritten symbols.
    """

    # system modules are missing __file__ attribute, and checking by
    # id is insufficient to prevent infinite loops, hence forbid missing
    # __file__
    if not get_symbol_file(original_module):
      self._done_modules[original_module] = original_module

    if original_module in self._done_modules:
      return self._done_modules[original_module]

    self._module_stack.append(original_module.__file__)

    updated_symbols = {}  # symbols that got touched

    for symbol_name, symbol in original_module.__dict__.items():

      # Case 1: symbol is directly replaced by symbol_rewriter
      new_symbol = self.symbol_rewriter(symbol)
      if new_symbol:
        updated_symbols[symbol_name] = new_symbol
        print("Rewrote symbol %s in %s" % (symbol_name,
                                           original_module.__name__))

      # Case 2: symbol is a module which may be affected by symbol_rewriter
      elif isinstance(symbol, types.ModuleType):
        if get_symbol_file(symbol) not in self._module_stack:
          new_symbol = self._rewrite_module(symbol)

          if new_symbol.__name__ != symbol.__name__:
            updated_symbols[symbol_name] = new_symbol
            print("Replaced %s in %s from %s" % (symbol_name,
                                                 original_module.__name__,
                                                 symbol.__name__))

      # Case 3: symbol is defined in a module which may be affected
      # by symbol rewriter
      elif hasattr(symbol, "__module__"):
        if symbol.__module__ != original_module.__name__:
          symbol_file = get_symbol_file(symbol)
          if symbol_file and symbol_file not in self._module_stack:
            symbol_module = sys.modules[symbol.__module__]
            new_symbol_module = self._rewrite_module(symbol_module)

            if new_symbol_module.__name__ != symbol_module.__name__:
              updated_symbols[symbol_name] = new_symbol_module.__dict__[
                  symbol.__name__]

    # nothing was modified, so return module unchanged
    if not updated_symbols:
      self._done_modules[original_module] = original_module
      self._module_stack.pop()
      return original_module


    new_module_name = self.module_prefix + original_module.__name__
    new_module = imp.new_module(new_module_name)
    new_module.__package__ = ""
    new_module.__file__ = self.module_prefix + original_module.__file__

    for symbol_name, symbol in original_module.__dict__.items():

      # don't rewrite new module attributes that we just set
      if symbol_name in ('__file__', '__name__', '__package__'):
        continue

      if symbol_name in updated_symbols:
        new_symbol = updated_symbols[symbol_name]
        if (hasattr(new_symbol, "__module__") and
            new_symbol.__module__ == original_module.__name__):
          new_symbol.__module__ = new_module.__name__

        new_module.__dict__[symbol_name] = new_symbol

      # it's a function whose definition wasn't updated
      elif isinstance(symbol, types.FunctionType):
        # if it's a function in current module, copy it to update its globals
        if symbol.__module__ == original_module.__name__:
          new_symbol = copy_function(symbol, new_module)
        # otherwise retain old reference
        else:
          new_symbol = symbol
        new_module.__dict__[symbol_name] = new_symbol

      else: # objects, classes, constants remain unchanged
        new_module.__dict__[symbol_name] = symbol

    sys.modules[new_module_name] = new_module
    self._done_modules[original_module] = new_module
    self._module_stack.pop()
    return new_module

