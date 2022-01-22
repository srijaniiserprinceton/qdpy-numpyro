import argparse
import numpy as np
from tqdm import tqdm
import sys
import os
from scipy import sparse

from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
# enabling 64 bits and logging compilatino
config.update("jax_log_compiles", 0)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
outdir = f"{scratch_dir}/qdpy_jax"
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
# importing local package 
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse
#-----------------------------------------------------------------#
GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults,
                            relpath=outdir)

__, GVARS_TR, __ = GVARS.get_all_GVAR()
#-----------------------------------------------------------------#
# precomputing the perform tests and checks and generate true synthetic eigvals
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sp_indices_all =\
                                precompute.build_hypmat_all_cenmults()

#----------------miscellaneous parameters-------------------------#
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.ctrl_arr_dpt_clipped.shape[0]  # number of s
dim_hyper = int(np.loadtxt('.dimhyper'))

#-----------------------------------------------------------------# 
# storing the true parameters and flattening appropriately                                   
true_params = 1.* GVARS.ctrl_arr_dpt_clipped
true_params_flat = np.reshape(true_params, (len_s * GVARS.nc), 'F')

# reshaping to make dotting seamless
param_coeff = np.reshape(noc_hypmat_all_sparse,
                         (nmults, len_s * GVARS.nc, -1), 'F')

#-----------------------------------------------------------------#

def model():
    eigval_model = np.array([])
    
    pred_hypmat_all_sparse = true_params_flat @ param_coeff +\
                             fixed_hypmat_all_sparse
    
    for mult_ind in tqdm(range(nmults), desc="Solving eigval problem..."):
        eigval_mult = np.zeros(2*ellmax+1)
        # converting to dense
        hypmat = sparse.coo_matrix((pred_hypmat_all_sparse[mult_ind],
                                    sp_indices_all[mult_ind])).toarray()

        # solving the eigenvalue problem and mapping eigenvalues
        ell0 = ell0_arr[mult_ind]
        omegaref = omega0_arr[mult_ind]
        
        eigval_qdpt_mult = get_eigs(hypmat)[:2*ell0+1]/2./omegaref
        eigval_qdpt_mult *= GVARS.OM*1e6
        eigval_mult[:2*ell0+1] = eigval_qdpt_mult
        
        # storing the correct order of nmult
        eigval_model = np.append(eigval_model,
                                 eigval_mult)
                       

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

#-----------------------------------------------------------------#

if __name__ == "__main__":
    ellmax = np.max(GVARS.ell0_arr)
    # model_ = jit(model)
    # eigvals_true = compare_hypmat()
    eigvals_true = model()

    # storing the eigvals sigmas
    eigvals_sigma = np.ones_like(eigvals_true)

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
    np.save(f"{outdir}/eigvals_model.npy", eigvals_true) 
    np.save(f'{outdir}/eigvals_sigma_model.npy', eigvals_sigma)
    
    # saving the HMI acoeffs and their uncertainties
    np.save(f'{outdir}/acoeffs_sigma_HMI.npy', GVARS.acoeffs_sigma)
    np.save(f'{outdir}/acoeffs_HMI.npy', GVARS.acoeffs_true)
    
    
