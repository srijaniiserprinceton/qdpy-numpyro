import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse
import os
import sys

import jax
from jax import random
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.lax import fori_loop as foril
from jax.ops import index_update as jidx_update
from jax.experimental import sparse as jsparse
from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)
# config.update("jax_log_compiles", 1)
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

#----------------------import custom packages------------------------#
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse
#---------------------------------------------------------------------# 
GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults,
                            relpath=outdir)

__, GVARS_TR, __ = GVARS.get_all_GVAR()
#-------------------parameters to be inverted for--------------------# 
# the indices of ctrl points that we want to invert for
ind_min, ind_max = 0, GVARS.knot_num-1
cind_arr = np.arange(ind_min, ind_max + 1)

# the angular degrees we want to invert for
smin, smax = 1, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.arange(smin_ind, smax_ind+1)
#---------------------------------------------------------------------#

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sp_indices_all =\
    precompute.build_hypmat_all_cenmults()

# making them array-like
noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)
ell0_arr = np.asarray(ell0_arr)
omega0_arr = np.asarray(omega0_arr)
sp_indices_all = np.asarray(sp_indices_all).astype('int')

#---------------computing miscellaneous shape parameters---------------#
len_data = len(omega0_arr)
nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)
dim_hyper = int(np.loadtxt('.dimhyper'))
len_hyper_arr = fixed_hypmat_all_sparse.shape[-1]
ellmax = np.max(GVARS.ell0_arr)

#---------section to add some more ctrl points to fixed part------------#
# array of ctrl points which aer non-zero for the fixed splines
c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
# filling in with the dpt wsr by default 
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in range(smin_ind, smax_ind+1):
    for cind in cind_arr:
        c_fixed[sind, cind] = 0.0
        c_fixed[sind, cind] = 0.0

# this is the modified fixed part of the supermatrix
fixed_hypmat_sparse = np.zeros((nmults, len_hyper_arr))

# looping over multiplets to precompute modified fixed part
for i in range(nmults):
    _fixmat = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                               fixed_hypmat_all_sparse[i],
                                               c_fixed, nc, len_s)

    fixed_hypmat_sparse[i, :] = _fixmat

#---------storing the true parameters which are allowed to vary-------# 
true_params = np.zeros((smax_ind - smin_ind + 1, ind_max - ind_min + 1))

for si, sind in enumerate(sind_arr):
    for ci, cind in enumerate(cind_arr):
        true_params[si, ci] = GVARS.ctrl_arr_dpt_clipped[sind, cind]


#------------------flattening for easy dot product--------------------#
true_params_flat = np.reshape(true_params,
                              (len(sind_arr) * len(cind_arr)), 'F')

param_coeff = np.zeros((nmults, len(sind_arr), len(cind_arr), len_hyper_arr))

for i in range(nmults):
    for si, sind in enumerate(sind_arr):
        for ci, cind in enumerate(cind_arr):
            param_coeff[i, si, ci, :] = noc_hypmat_all_sparse[i, sind, cind, :]

# flattening in (len_s x cind) dimensions to facilitate seamless dotting
param_coeff_flat = np.reshape(param_coeff,
                              (nmults, len(sind_arr) * len(cind_arr), -1), 'F')
del param_coeff
#---------------------------------------------------------------------#
# converting to array and changing the axes to nmult X element_idx X xy-identifier
hypmat_idx = np.moveaxis(sp_indices_all, 1, -1)

#----------------saving precomputed parameters------------------------#
np.save(f'{outdir}/true_params_flat.npy', true_params_flat)
np.save(f'{outdir}/param_coeff_flat.npy', param_coeff_flat)
np.save(f'{outdir}/fixed_part.npy', fixed_hypmat_sparse)
np.save(f'{outdir}/sparse_idx.npy', hypmat_idx)
np.save(f'{outdir}/omega0_arr.npy', omega0_arr)
np.save(f'{outdir}/ell0_arr.npy', ell0_arr)
np.save(f'{outdir}/cind_arr.npy', cind_arr)
np.save(f'{outdir}/sind_arr.npy', sind_arr)

# sys.exit()
#-----------------------------------------------------------------#
synth_hypmat_sparse = true_params_flat @ param_coeff_flat + fixed_hypmat_sparse

def model():
    eigval_model = np.array([])

    for mult_ind in tqdm(range(nmults), desc="Solving eigval problem..."):
        eigval_mult = np.zeros(2*ellmax+1)
        # converting to dense
        hypmat = sparse.coo_matrix((synth_hypmat_sparse[mult_ind],
                                    sp_indices_all[mult_ind]),
                                   shape=(dim_hyper, dim_hyper)).toarray()

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

#--------saving miscellaneous files of eigvals and acoeffs---------#                      
eigvals_true = model()

# saving the synthetic eigvals and HMI acoeffs
np.save(f"{outdir}/data_model.npy", eigvals_true)
np.save(f'{outdir}/acoeffs_HMI.npy', GVARS.acoeffs_true)
np.save(f'{outdir}/acoeffs_sigma_HMI.npy', GVARS.acoeffs_sigma)

sys.exit()

#-------------COMPARING AGAINST supmat_qdpt and dpy_jax----------------#
# testing only valid for nmin = 0, nmax = 0, lmin = 200, lmax = 201
synth_hypmat = np.zeros((nmults, dim_hyper, dim_hyper))

DPT_eigvals_from_qdpy= np.zeros(2 * (2*201 + 1))

start_idx = 0
for i in range(2):    
    ell0 = ell0_arr[i]
    end_idx  = start_idx + (2*ell0 + 1)
    # densifying
    synth_hypmat[i] = sparse.coo_matrix((synth_hypmat_sparse[i], sp_indices_all[i]),
                                        shape = (dim_hyper, dim_hyper)).toarray()

    # DPT eigenvalues in muHz
    DPT_eigvals_from_qdpy[start_idx:end_idx] =\
        np.diag(synth_hypmat[i])[:2*ell0+1] * 1.0/2./omega0_arr[i]*GVARS.OM*1e6
    
    start_idx += 2*201 + 1
 
#----------------------COMPARING AGAINST supmat_qdpt.npy-----------------#

for i, ell0 in enumerate(ell0_arr):
    # qdpt supermatrix from qdPy
    qdpt_supmat = np.load(f'supmat_qdpt_{ell0}.npy').real
    dim_super = qdpt_supmat.shape[0]
    np.testing.assert_array_almost_equal(qdpt_supmat,
                                         synth_hypmat[i][:dim_super,:dim_super])


#-----------------------COMPARING AGAINST dpy_jax--------------------------#
eigvals_from_dpy_jax = np.load(f'{outdir}/eigvals_model_dpy_jax.npy')
np.testing.assert_array_almost_equal(DPT_eigvals_from_qdpy,
                                     eigvals_from_dpy_jax)
