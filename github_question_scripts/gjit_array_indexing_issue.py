from functools import partial
from collections import namedtuple
from jax import jit
import jax.numpy as jnp
import numpy as np

def some_hash_function(x):
  return int(jnp.sum(x))

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

def gnool_jit(fun, static_array_argnums=()):
  @partial(jit, static_argnums=static_array_argnums)
  def callee(*args):
    args = list(args)
    for i in static_array_argnums:
      if isinstance(args[i], tuple):
        args[i] = args[i].__class__(*[a.val for a in args[i]])
    else:
        args[i] = args[i].val
    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      if isinstance(args[i], tuple):
        all_as = [HashableArrayWrapper(a) for a in args[i]]
        args[i] = args[i].__class__(*all_as)
    else:
        args[i] = HashableArrayWrapper(args[i])
    return callee(*args)

  return caller


def func(dim_array):
    ret_arr = jnp.zeros(dim_array[-1] - dim_array[0])
    return ret_arr


dim_array_jnp = jnp.array([1,5], dtype='int32')
dim_array_np = np.array(dim_array_jnp)

func_ = gnool_jit(func, static_array_argnums=(0,))

__ = func_(dim_array_np)

__ = func_(dim_array_jnp)
