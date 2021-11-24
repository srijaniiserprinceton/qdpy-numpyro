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
import jax_functions as jf
import globalvars as gvar_jax
import sparse_precompute_acoeff as precompute
import build_hypermatrix_sparse as build_hm_sparse

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
__, GVARS_TR, __ = GVARS.get_all_GVAR()
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.wsr.shape[0]  # number of s

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                precompute.build_hypmat_all_cenmults()

def model():
    # building the entire hypermatrix
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  GVARS.ctrl_arr_dpt_clipped,
                                                  GVARS.nc, len_s)
    
    # finding the eigenvalues of hypermatrix
    diag_dense = diag_evals.todense()
    return diag_dense


def eigval_sort_slice(eigval, eigvec):
    """Sorts eigenvalues using the eigenvectors"""
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = jnp.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)
    return eigval[eigbasis_sort]


def get_eigs(mat):
    """Returns the sorted eigenvalues of a real symmetric matrix"""
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals

def compare_hypmat():
    diag = model_().block_until_ready()
    # return diag
    import matplotlib.pyplot as plt
    import numpy as np
    # plotting difference with qdpt.py
    supmat_qdpt = np.load("supmat_qdpt_200.npy").real

    sm1 = diag
    sm2 = np.diag(supmat_qdpt)[:401]

    plt.figure(figsize=(10, 5))
    plt.plot(sm1 - sm2)
    print(f"Max diff = {abs(sm1 - sm2).max():.3e}")
    plt.savefig('supmat_diff.pdf')
    return diag

if __name__ == "__main__":
    model_ = jit(model)
    # eigvals_true = compare_hypmat()
    eigvals_true = model_()
    print(f"num elements = {len(eigvals_true)}")
    np.save("evals_model.npy", eigvals_true/2./omega0_arr*GVARS.OM*1e6)

    # storing the eigvals sigmas
    eigvals_sigma = np.ones_like(eigvals_true)

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

    np.save('eigvals_sigma.npy', eigvals_sigma)
