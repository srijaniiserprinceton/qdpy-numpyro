from functools import partial
from collections import namedtuple
from jax import jit
import jax.numpy as jnp

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
      args[i] = args[i].val
    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = HashableArrayWrapper(args[i])
    return callee(*args)

  return caller


###


@partial(gnool_jit, static_array_argnums=(0,1))
def f(a,x,y):
  print('re-tracing f!')
  return a * (x ** 2 + y ** 2)


@partial(gnool_jit, static_array_argnums=(0,1))
def g(nt,x,y):
  print('re-tracing g!')
  a = nt.a
  return a * (x ** 2 + y ** 2)

x = jnp.array([1,2,3])
y = jnp.array([4,5,6])

a = 1

__ = f(a,x,y)
__ = f(a,x,y)

nt = namedtuple('nt','a')
nt = nt(a)

__ = g(nt,x,y)
__ = g(nt,x,y)
