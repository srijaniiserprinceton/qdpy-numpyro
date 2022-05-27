import argparse
import numpy as np
import sys
import os

from jax import jit
from jax.config import config
# enabling 64 bits and logging compilatino
# config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

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
parser.add_argument("--smax_global", help="smax of the wsr to read",
                    type=int, default=19)
parser.add_argument("--knot_num", help="number of knots beyond rth",
                    type=int, default=5)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
parser.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
parser.add_argument("--tslen", help="72d or 360d",
                    type=int, default=72)
parser.add_argument("--daynum", help="day from MDI epoch",
                    type=int, default=6328)
parser.add_argument("--numsplits", help="number of splitting coefficients",
                    type=int, default=18)
parser.add_argument("--batch_run", help="flag to indicate its a batch run",
                    type=int, default=0)
parser.add_argument("--batch_rundir", help="local directory for batch run",
                    type=str, default=".")
parser.add_argument("--compare_qdPy", help="compare with qdpy",
                    action="store_true")
ARGS = parser.parse_args()
#------------------------------------------------------------------------# 
if(not ARGS.batch_run):
    n0lminlmax_dir = f"{package_dir}/dpy_jax"
    outdir = f"{scratch_dir}/dpy_jax"
else:
    n0lminlmax_dir = f"{ARGS.batch_rundir}"
    outdir = f"{ARGS.batch_rundir}"

with open(f"{n0lminlmax_dir}/.n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.rth}" + "\n" +
            f"{ARGS.knot_num}" + "\n" +
            f"{ARGS.load_mults}" + "\n" + 
            f"{ARGS.tslen}" + "\n" + 
            f"{ARGS.daynum}" + "\n" +
            f"{ARGS.numsplits}" + "\n" +
            f"{ARGS.smax_global}")
    
#-----------------------------------------------------------------#
# importing local package 
from qdpy import globalvars as gvar_jax
from qdpy import build_hypermatrix_sparse as build_hm_sparse
#-----------------------------------------------------------------#
GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            smax_global=ARGS.smax_global,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults,
                            relpath=n0lminlmax_dir,
                            instrument=ARGS.instrument,
                            tslen=ARGS.tslen,
                            daynum=ARGS.daynum,
                            numsplits=ARGS.numsplits)

# the only import which needs .n0-lmin-lmax.dat to be created beforehand
if(not ARGS.batch_run):
    import sparse_precompute_acoeff as precompute
else:
    sys.path.append(f"{ARGS.batch_rundir}")
    import sparse_precompute_acoeff_batch as precompute
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
    evals_dpy = model_().block_until_ready()[403:804]
    
    # compraring with qdpt.py
    supmat_qdpt_200 = np.load("supmat_qdpt_200_9.npy").real

    evals_qdpt = np.diag(supmat_qdpt_200)[:401]
    print(f"from qdPy   : {evals_qdpt[0]:.6e}")
    print(f"from dpy-jax: {evals_dpy[0]:.6e}")

    # comparing with qdpt.py
    np.testing.assert_array_almost_equal(evals_dpy, evals_qdpt)
    
    print('Eigval test with qdPy passed')

    return evals_dpy

#-----------------------------------------------------------------#

if __name__ == "__main__":
    model_ = jit(model)
    if ARGS.compare_qdPy: eigvals_true = compare_hypmat()
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
                        GVARS.eigvals_sigma[start_ind_gvar:end_ind_gvar]
        
        start_ind +=  2 * ellmax + 1
        start_ind_gvar += 2 * ell + 1

    #--------saving miscellaneous files of eigvals and acoeffs---------#
    # saving the synthetic eigvals and their uncertainties
    sfx = GVARS.filename_suffix
            
    np.save(f"{outdir}/eigvals_model.{sfx}.npy", eigvals_true/2./omega0_arr*GVARS.OM*1e6)
    np.save(f'{outdir}/eigvals_sigma_model.{sfx}.npy', eigvals_sigma)
    
    # saving the HMI acoeffs and their uncertainties
    np.save(f'{outdir}/acoeffs_sigma_HMI.{sfx}.npy', GVARS.acoeffs_sigma)
    np.save(f'{outdir}/acoeffs_HMI.{sfx}.npy', GVARS.acoeffs_true)

    #-----------------------------------------------------------------#
