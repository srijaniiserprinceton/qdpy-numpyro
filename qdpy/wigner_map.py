from tqdm import tqdm
import numpy as np
import py3nj
import time

import jax
import jax.numpy as jnp
from jax.lax import fori_loop as foril
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)


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


def get_wigners_dpy(nl_nbs, s_arr, wig_list, wig_idx):
    # re-converting it back to array
    nl_nbs = jnp.asarray(nl_nbs)
    
    @np.vectorize
    def find_idx_fac(ell1, s, ell2, m):
        dell = abs(ell1 - ell2)
        ell = min(ell1, ell2)
        idx1 = ell*(ell+1)//2 + abs(m)
        idx2 = s*(s+1)//2 + dell
        fac = 1

        if m < 0:
            fac = -1

        max_ord_mag_idx1 = 5
        wig_idx = idx2*(10**max_ord_mag_idx1) + idx1
        return wig_idx, fac

    num_multiplets = nl_nbs.shape[0]
    num_blocks = int(num_multiplets**2)

    for i in tqdm(range(num_multiplets), desc=f"Precomputing wigners..."):
        ell = np.array([nl_nbs[i, 1]])[0]
        m = np.arange(0, ell+1)
        larr = np.ones_like(m)*ell
        for s in s_arr:
            widx, fac = find_idx_fac(larr, s, larr, m)
            exists = True
            
            try:
                _i1 = wig_idx.index(widx)
                exists = True
            except ValueError:
                exists = False
                
            if not exists:
                wigvals = w3j_vecm(ell, s, ell, -m, 0*m, m)
                wig_list.extend(list(wigvals))
                wig_idx.extend(list(widx))

    return wig_list, wig_idx


def get_wigners_qdpy(nl_nbs, s_arr, wig_list, wig_idx):
    # re-converting it back to array
    nl_nbs = jnp.asarray(nl_nbs)
    
    @np.vectorize
    def find_idx_fac(ell1, s, ell2, m):
        dell = abs(ell1 - ell2)
        ell = min(ell1, ell2)
        idx1 = ell*(ell+1)//2 + abs(m)
        idx2 = s*(s+1)//2 + dell
        fac = 1

        if m < 0:
            fac = -1

        max_ord_mag_idx1 = 5
        wig_idx = idx2*(10**max_ord_mag_idx1) + idx1
        return wig_idx, fac

    num_multiplets = nl_nbs.shape[0]
    num_blocks = int(num_multiplets**2)

    for i in range(num_multiplets):
        for ii in range(i, num_multiplets):
            ell1 = np.array([nl_nbs[i, 1]])[0]
            ell2 = np.array([nl_nbs[ii, 1]])[0]
            ellmin = min(ell1, ell2)
            m = np.arange(0, ellmin+1)
            l1arr = np.ones_like(m)*ell1
            l2arr = np.ones_like(m)*ell2
            for s in s_arr:
                dell = abs(ell2 - ell1)
                widx, fac = find_idx_fac(l1arr, s, l2arr, m)
                exists = True

                try:
                    _i1 = wig_idx.index(widx)
                    exists = True
                except ValueError:
                    exists = False

                if not exists:
                    wigvals = w3j_vecm(ell1, s, ell2, -m, 0*m, m)
                    wig_list.extend(list(wigvals))
                    wig_idx.extend(list(widx))

    return wig_list, wig_idx

def issorted(a):
    '''function to check if the elements of a 1D array are sorted'''
    return jnp.all(a[:-1] <= a[1:])


def ind2sub(cont_ind, nrows, ncols):
    '''function to find 2-d index for a contiguous numbering
    of elements in a matrix'''
    return cont_ind//nrows, cont_ind%ncols


def find_idx(ell1, s, ell2, m):
    '''New method for specific use-case of qdPy
       /ell1 s ell2\
       \-|m| 0 |m| /
    
    Inputs:
    ------
    ell1 - int
    s - int
    ell2 - int
    m - int
    All are parameters as described in the matrix above

    Returns:
    --------
    wig_idx - int
        index of the wigner3j symbol
    fac - float
        sign of wigner
    '''
    fac = np.sign(m)
    ell = np.minimum(ell1, ell2)
    dell = np.abs(ell1 - ell2)
    idx1 = ell*(ell+1)//2 + np.abs(m)
    idx2 = s*(s+1)//2 + dell

    # computing a unified index for the wigner
    max_ord_mag_idx1 = 5
    wig_idx = idx2*(10**max_ord_mag_idx1) + idx1
    return wig_idx, fac 


def foril_func(i):
    return i, _find_idx(ell1, s, ell2, m)


def get_wig_from_pc(ell1, s, ell2, m):
    wig1 = w3j_vecm(ell1, s, ell2, -m, 0, m)
    idx, fac = find_idx(ell1, s, ell2, m)
    wigidx_local = jnp.searchsorted(wig_idx, idx)
    wig2 = fac * wig_list[wigidx_local]
    tv = np.isclose(wig1, wig2)
    print(f'Match = {tv}')
    return wig1, wig2


def compute_uniq_wigners(ell, s, ellp, m):
    wig_idx, fac = _find_idx(ell, s, ellp, m)
    wig_list = w3j_vecm(ell, s, ellp, -m, 0*m, m)

    sortind_wig_idx = np.argsort(wig_idx, kind='quicksort')
    wig_idx = wig_idx[sortind_wig_idx]
    wig_list = wig_list[sortind_wig_idx]
    return wig_list, wig_idx


if __name__ == "__main__":
    _find_idx = jax.jit(find_idx)
    ell1, s, ell2 = 200, 5, 202
    m = jnp.arange(ell1+1)
    wig_list, wig_idx = compute_uniq_wigners(ell1, s, ell2, m)
    wig_list = jnp.asarray(wig_list)
    m_test = np.arange(16)
    __ = get_wig_from_pc(ell1, s, ell2, m_test)
    __ = get_wig_from_pc(ell2, s, ell1, m_test)
    __ = get_wig_from_pc(ell1, s, ell2, -m_test)
    __ = get_wig_from_pc(ell2, s, ell1, -m_test)

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
    for __ in range(Niter): idx, fac = _find_idx(ell1, s, ell2, m)
    t4 = time.time()

    print(f'Time taken for a 1.2 billion computations in hours:' +
          f' {(t4-t3) / Niter * 1.2e9 / 3600:.2f}')
