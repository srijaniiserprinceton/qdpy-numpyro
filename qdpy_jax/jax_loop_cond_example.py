import jax
import jax.numpy as jnp
import time

def pure_uses_internal_state(x):
    state = dict(even=0, odd=0)
    for i in range(10):
        state['even' if i % 2 == 0 else 'odd'] += x
    return state['even'] + state['odd']


def jax_pure_uses_internal_state(x):
    state = jnp.array([0,0],dtype='float32')

    def f_state(i, state):
        true_fun = lambda state: jax.ops.index_update(state,0,state[0]+x)
        false_fun = lambda state: jax.ops.index_update(state,1,state[1]+x)

        return jax.lax.cond(i % 2 == 0, true_fun, false_fun, state)

    state = jax.lax.fori_loop(0,10,f_state,state)

    return state[0] + state[1]

# testing if they return the same value
print("Original function returns: ", pure_uses_internal_state(5.))
print("JAX function returns: ", jax_pure_uses_internal_state(5.))

Niter = 1000

t1 = time.time()
for i in range(Niter): __ = jax_pure_uses_internal_state(5.)
t2 = time.time()

# jitting
_jax_pure_uses_internal_state = jax.jit(jax_pure_uses_internal_state)
# compiling jitted function by running once
__ = _jax_pure_uses_internal_state(5.)

t3 = time.time()
for i in range(Niter): __ = _jax_pure_uses_internal_state(5.).block_until_ready()
t4 = time.time()

print("Time in seconds non-jitted: ", (t2-t1)/Niter)
print("Time in seconds jitted: ", (t4-t3)/Niter)
