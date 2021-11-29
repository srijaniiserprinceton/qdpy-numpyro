import argparse
import numpy as np
from scipy import sparse

from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
import sys

# enabling 64 bits and logging compilatino
config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--rth", help="threshold radius",
                    type=float, default=0.98)
parser.add_argument("--knot_num", help="number of knots beyond rth",
                    type=int, default=10)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.rth}" + "\n" +
            f"{ARGS.knot_num}" + "\n" +
            f"{ARGS.load_mults}")

# importing local package 
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
__, GVARS_TR, __ = GVARS.get_all_GVAR()
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.wsr.shape[0]  # number of s
np.save('acoeffs_sigma.npy', GVARS.acoeffs_sigma)
np.save('acoeffs_true.npy', GVARS.acoeffs_true)


noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sp_indices_all =\
                                precompute.build_hypmat_all_cenmults()

# converting to numpy ndarrays from lists
noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)

def model():
    eigval_model = jnp.array([])
    
    for i in range(nmults-1, -1, -1):
        hypmat_sparse = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                         fixed_hypmat_all_sparse[i],
                                                         GVARS.ctrl_arr_dpt_clipped,
                                                         GVARS.nc, len_s)

        # converting to dense
        hypmat = sparse.coo_matrix((hypmat_sparse, sp_indices_all[i])).toarray()

        # solving the eigenvalue problem and mapping eigenvalues
        ell0 = ell0_arr[i]
        omegaref = omega0_arr[i]
        eigval_qdpt_mult = get_eigs(hypmat)[:2*ell0+1]/2./omegaref
        eigval_qdpt_mult *= GVARS.OM*1e6

        # storing the correct order of nmult
        eigval_model = jnp.append(eigval_qdpt_mult, eigval_model)

    return eigval_model

def eigval_sort_slice(eigval, eigvec):
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = jnp.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)
    return eigval[eigbasis_sort]


def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals

def get_eigvals_sigma(len_evals_true):
    '''Function to get the sigma from data
    for the frequency splittings.
    '''
    # storing the eigvals sigmas                                                                                                                             
    eigvals_sigma = jnp.array([])

    ellmax = np.max(GVARS.ell0_arr)

    start_ind_gvar = 0
    start_ind = 0

    for i, ell in enumerate(GVARS.ell0_arr):
        end_ind_gvar = start_ind_gvar + 2 * ell + 1

        # storing in the correct order of nmult
        eigvals_sigma = jnp.append(eigvals_sigma,
                                   GVARS_TR.eigvals_sigma[start_ind_gvar:end_ind_gvar])

        start_ind_gvar += 2 * ell + 1

    return eigvals_sigma

if __name__ == "__main__":
    # model_ = jit(model)
    # eigvals_true = compare_hypmat()
    eigvals_true = model()
    eigvals_sigma = get_eigvals_sigma(len(eigvals_true))
    print(f"num elements = {len(eigvals_true)}")
    np.save("evals_model.npy", eigvals_true) 
    np.save('eigvals_sigma.npy', eigvals_sigma)
    np.save('acoeffs_sigma.npy', GVARS.acoeffs_sigma)
    np.save('acoeffs_true.npy', GVARS.acoeffs_true)
