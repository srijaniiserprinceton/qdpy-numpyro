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


###

@partial(gnool_jit, static_array_argnums=(0,1,2))
def f(a,x,y):
  print('re-tracing f!')
  return a * (x ** 2 + y ** 2)


@partial(gnool_jit, static_array_argnums=(0,1,2,3))
def g(nt,nt2,x,y):
  print('re-tracing g!')
  a = nt.a
  b = nt2.a
  return a * (x ** 2 + y ** 2) + b

x = jnp.array([1,2,3])
y = jnp.array([4,5,6])

a = 1

__ = f(a,x,y)
__ = f(a,x,y)

nt = namedtuple('nt','a b')
nt2 = namedtuple('nt2','a b c')
ntval = nt(a, a+1)
ntval2 = nt2(a, a+1, a+2)

print(g(ntval, ntval2,x,y))
print(g(ntval, ntval2,x,y))
