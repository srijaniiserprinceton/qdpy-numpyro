import jax
import jax.numpy as jnp

def func(ell):
    ell2 = jnp.add(ell, 1)
    print(ell2)
    m_arr = jnp.arange(-ell2, ell2+1, dtype='int32')
    return m_arr

# creating and compiling the jitted function
_func = jax.jit(func, static_argnums=(0,))
__ = _func(5).block_until_ready()

# running for different value of the static argument
for ell in range(1, 10):
    m_arr = _func(ell).block_until_ready()
    print(m_arr)
