import jax.numpy as jnp
from jax.lax import fori_loop as foril
import jax
import time

jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform) 

# function to check if the elements of a 1D array are sorted
def issorted(a):
    return jnp.all(a[:-1] <= a[1:])

# function to find 2-d index for a contiguous numbering
# of elements in a matrix
def ind2sub(cont_ind, nrows, ncols):
    return cont_ind//nrows, cont_ind%ncols

@jnp.vectorize
def find_idx(ell1, s, ell2, m):
    # New method for specific use-case of qdPy
    # /ell1 s ell2\
    # \ m  0  -m /

    def true_func(ell12m):
        ell1, ell2, m = ell12m
        fac = jax.lax.cond(m<0,
                           lambda fac: -1,
                           lambda fac: 1,
                           operand=None)
        dell = ell2 - ell1
        return ell1, dell, fac

    def false_func(ell12m):
        ell1, ell2, m = ell12m
        fac = jax.lax.cond(m>=0,
                           lambda fac: -1,
                           lambda fac: 1,
                           operand=None)
        dell = ell1 - ell2
        return ell2, dell, fac

    ell, dell, fac = jax.lax.cond(ell2>ell1,
                                  true_func,
                                  false_func,
                                  operand=(ell1, ell2, m))

    idx1 = ell*(ell+1)//2+jnp.abs(m)
    idx2 = s*(s+1)//2+dell

    return idx1, idx2, fac

def foril_func(i):
    return i, _find_idx(ell1, s, ell2, m)


# timing the functions with and without jitting
if __name__ == "__main__":
    # wigner parameters
    ell1, s, ell2 = 12, 3, 10
    # m = -9
    m = jnp.arange(ell1)
    ell1 = ell1*jnp.ones_like(m)
    s = s*jnp.ones_like(m)
    ell2 = ell2*jnp.ones_like(m)

    # timing the functions with and without jitting

    # timing the functions with and without jitting
    Niter = 10000

    # timing the unjitted version
    c = find_idx(ell1, s, ell2, m)

    
    # timing the jitted version
    _find_idx = jax.jit(find_idx)
    __ = _find_idx(ell1, s, ell2, m)

    
    t3 = time.time()
    for __ in range(Niter): idx1, idx2, fac = _find_idx(ell1, s, ell2, m)
    t4 = time.time()

    print('Time taken for a 1.2 billion computations in hours:',
          (t4-t3) / Niter * 1.2e9 / 3600)
