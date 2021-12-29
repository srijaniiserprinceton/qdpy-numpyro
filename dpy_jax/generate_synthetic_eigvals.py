import argparse
import numpy as np
import sys

from jax import jit
from jax.config import config
# enabling 64 bits and logging compilatino
config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

# importing local package 
from qdpy_jax import globalvars as gvar_jax
import sparse_precompute_acoeff as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

#-----------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--rth", help="threshold radius",
                    type=float, default=0.97)
parser.add_argument("--knot_num", help="number of knots beyond rth",
                    type=int, default=5)
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
#-----------------------------------------------------------------#

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
__, GVARS_TR, __ = GVARS.get_all_GVAR()

#-----------------------------------------------------------------#

# precomputing the perform tests and checks and generate true synthetic eigvals
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                precompute.build_hypmat_all_cenmults()

#-----------------------------------------------------------------#

len_s = GVARS.wsr.shape[0]  # number of s
def model():
    # building the entire hypermatrix
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  GVARS.ctrl_arr_dpt_clipped,
                                                  GVARS.nc, len_s)
    
    # finding the eigenvalues of hypermatrix
    diag_dense = diag_evals.todense()
    return diag_dense


def compare_hypmat():
    # generating the true synthetic eigenfrequencies
    evals_dpy = model_().block_until_ready()
    
    # compraring with qdpt.py
    supmat_qdpt_200 = np.load("supmat_qdpt_200.npy").real

    evals_qdpt = np.diag(supmat_qdpt_200)[:401]

    # comparing with qdpt.py
    np.testing.assert_array_almost_equal(evals_dpy, evals_qdpt)
    
    return evals_dpy

#-----------------------------------------------------------------#

if __name__ == "__main__":
    model_ = jit(model)
    # eigvals_true = compare_hypmat()
    eigvals_true = model_()

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

    #--------saving miscellaneous files of eigvals and acoeffs---------#
    # saving the synthetic eigvals and their uncertainties
    np.save("eigvals_model.npy", eigvals_true/2./omega0_arr*GVARS.OM*1e6)
    np.save('eigvals_sigma_model.npy', eigvals_sigma)
    
    # saving the HMI acoeffs and their uncertainties
    np.save('acoeffs_sigma_HMI.npy', GVARS.acoeffs_sigma)
    np.save('acoeffs_HMI.npy', GVARS.acoeffs_true)

    #-----------------------------------------------------------------#
