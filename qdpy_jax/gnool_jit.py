import jax.numpy as jnp
from jax import jit
from functools import partial

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
