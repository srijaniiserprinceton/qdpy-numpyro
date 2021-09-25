import jax.numpy as jnp
import jax
from collections import namedtuple

test_class = namedtuple('test_class', ['ell'])
    
def create_m_arr(test_class):
    print('Compiling ', test_class.ell)
    m_arr = jnp.arange(-test_class.ell, test_class.ell+1)
    return m_arr
    # return 2 * test_class.ell

# jitting function of the class
_create_m_arr = jax.jit(create_m_arr, static_argnums=(0,))

for ell in range(10,20):
    print('Executing ', ell)
    try_class = test_class(ell)
    __ = _create_m_arr(try_class).block_until_ready()
    # print(__)

for ell in range(10,30):
    print('Executing ', ell)
    try_class = test_class(ell)
    __ = _create_m_arr(try_class).block_until_ready()
    # print(__)
