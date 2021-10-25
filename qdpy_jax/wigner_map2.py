import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop as foril
import py3nj
import jax
import time

jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)


def w3j(l1, l2, l3, m1, m2, m3):
    """Computes the wigner-3j symbol for given l1, l2, l3, m1, m2, m3"""
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = int(2*m1)
    m2 = int(2*m2)
    m3 = int(2*m3)
    try:
        wigval = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    except ValueError:
        return 0.0
    return wigval


def w3j_vecm(l1, l2, l3, m1, m2, m3):
    """Computes the wigner-3j symbol for given l1, l2, l3, m1, m2, m3.

    Inputs:
    -------
    l1, l2, l3 - int
    m1, m2, m3 - np.ndarray(ndim=1, dtype=np.int32)

    Returns:
    --------
    wigvals - np.ndarray(ndim=1, dtype=np.float32)
    """
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    return wigvals


def get_wigners(nl_nbs, wig_list, idx1_list, idx2_list):
    @np.vectorize
    def find_idx_fac(ell1, s, ell2, m):
        dell = abs(ell1 - ell2)
        ell = min(ell1, ell2)
        idx1 = ell*(ell+1)//2 + abs(m)
        idx2 = s*(s+1)//2 + dell
        fac = 1
        if (ell2 > ell1) and (m < 0):
            fac = -1
        if (ell2 < ell1) and (m >= 0):
            fac = -1
        return idx1, idx2, fac

    num_multiplets = nl_nbs.shape[0]
    num_blocks = int(num_multiplets**2)
    s_arr = np.array([1, 3, 5])

    for i in range(num_multiplets):
        for ii in range(num_multiplets):
            ell1 = np.array([nl_nbs[i, 1]])[0]
            ell2 = np.array([nl_nbs[ii, 1]])[0]
            ellmin = min(ell1, ell2)
            m = np.arange(0, ellmin+1)
            l1arr = np.ones_like(m)*ell1
            l2arr = np.ones_like(m)*ell2
            for s in s_arr:
                dell = abs(ell2 - ell1)
                if s < dell:
                    continue

                idx1, idx2, fac = find_idx_fac(l1arr, s, l2arr, m)
                exists = True

                try:
                    _i1 = idx1_list.index(idx1[4])
                    _i2 = idx2_list.index(idx2[4])
                    mask1 = np.array(idx1_list) == idx1[4]
                    mask2 = np.array(idx2_list) == idx2[4]
                    exists = (mask1*mask2).sum().astype('bool')
                except ValueError:
                    exists = False

                if not exists:
                    print(f'{ell1} {s} {ell2}')
                    wigvals = w3j_vecm(ell1, s, ell2, -m, 0*m, m)
                    idx1_list.extend(list(idx1))
                    idx2_list.extend(list(idx2))
                    wig_list.extend(list(wigvals))
    return wig_list, idx1_list, idx2_list



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
    ell1, s, ell2 = 220, 5, 222
    # m = -9
    m = jnp.arange(-ell1, ell1+1)
    ell1 = ell1*jnp.ones_like(m)
    s = s*jnp.ones_like(m)
    ell2 = ell2*jnp.ones_like(m)

    # timing the functions with and without jitting
    Niter = 1000

    # timing the unjitted version
    c = find_idx(ell1, s, ell2, m)

    # timing the jitted version
    _find_idx = jax.jit(find_idx)
    __ = _find_idx(ell1, s, ell2, m)

    t3 = time.time()
    for __ in range(Niter): idx1, idx2, fac = _find_idx(ell1, s, ell2, m)
    t4 = time.time()

    print(f'Time taken for a 1.2 billion computations in hours:' +
          f' {(t4-t3) / Niter * 1.2e9 / 3600:.2f}')
