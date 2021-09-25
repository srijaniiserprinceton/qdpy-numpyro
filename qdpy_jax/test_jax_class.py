import jax.numpy as jnp
import jax

class test_class():
    def __init__(self, ell):
        self.ell = ell

    def create_m_arr(self, ell1):
        print('Compiling ', self.ell)
        m_arr = jnp.arange(-self.ell, self.ell+1)
        return m_arr

    
# creating instance of the class
try_class = test_class(10)

# jitting function of the class
_create_m_arr = jax.jit(try_class.create_m_arr, static_argnums=(0,))

for ell in range(10,20):
    print('Executing ', ell)
    try_class.ell = ell
    __ = _create_m_arr(ell).block_until_ready()

for ell in range(10,30):
    print('Executing ', ell)
    try_class.ell = ell
    __ = _create_m_arr(ell).block_until_ready()
