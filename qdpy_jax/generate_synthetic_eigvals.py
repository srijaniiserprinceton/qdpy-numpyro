import argparse
import numpy as np

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


noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr =\
                                precompute.build_hypmat_all_cenmults()


def model():
    eigval_model = jnp.array([])

    for i in range(nmults):
        diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                    fixed_hypmat_all_sparse[i],
                                                    GVARS.ctrl_arr_dpt_clipped,
                                                    GVARS.nc, len_s)

        ell0 = GVARS.ell0_arr[i]
        eigval_dpt_mult = jnp.diag(diag_evals.todense())
        eigval_dpt_mult *= 1.0/2./omega0_arr[i]*GVARS.OM*1e6
        eigval_dpt_mult = eigval_dpt_mult
        eigval_model = jnp.append(eigval_model, eigval_dpt_mult)

    return eigval_model

def get_eigvals_sigma(len_evals_true):
    '''Function to get the sigma from data
    for the frequency splittings.
    '''
    # storing the eigvals sigmas                                                                                                                                            
    eigvals_sigma = np.ones(len_evals_true)

    ellmax = np.max(GVARS.ell0_arr)

    start_ind_gvar = 0
    start_ind = 0

    for i, ell in enumerate(GVARS.ell0_arr):
        end_ind = start_ind + 2 * ell + 1
        end_ind_gvar = start_ind_gvar + 2 * ell + 1

        eigvals_sigma[start_ind:end_ind] *=\
                        GVARS_TR.eigvals_sigma[start_ind_gvar:end_ind_gvar]

        start_ind +=  2 * ellmax + 1
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
