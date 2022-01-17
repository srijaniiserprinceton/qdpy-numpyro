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
config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

#----------------------import custom packages------------------------#
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

#-------------------parameters to be inverted for--------------------# 
# the indices of ctrl points that we want to invert for
ind_min, ind_max = 0, 3
cind_arr = np.arange(ind_min, ind_max + 1)

# the angular degrees we want to invert for
smin, smax = 3, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.arange(smin_ind, smax_ind+1)
#---------------------------------------------------------------------#

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

GVARS_PATHS, GVARS_TR, __ = GVARS.get_all_GVAR()

#----------------------------------------------------------------------#

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
param_coeff = np.reshape(param_coeff,
                         (nmults, len(sind_arr) * len(cind_arr), -1), 'F')

#---------------------------------------------------------------------#
# converting to array and changing the axes to nmult X element_idx X xy-identifier
hypmat_idx = np.moveaxis(sp_indices_all, 1, -1)

#----------------saving precomputed parameters------------------------#
np.save('fixed_part.npy', fixed_hypmat_sparse)
np.save('param_coeff.npy', param_coeff)
np.save('sparse_idx.npy', hypmat_idx)
np.save('true_params.npy', true_params_flat)
np.save('omega0_arr.npy', omega0_arr)
np.save('ell0_arr.npy', ell0_arr)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)

# sys.exit()

#-------------COMPARING AGAINST supmat_qdpt and dpy_jax----------------#
# testing only valid for nmin = 0, nmax = 0, lmin = 200, lmax = 201
synth_hypmat = np.zeros((nmults, dim_hyper, dim_hyper))

DPT_eigvals_from_qdpy= np.zeros(2 * (2*201 + 1))

synth_hypmat_sparse = true_params_flat @ param_coeff + fixed_hypmat_sparse

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
eigvals_from_dpy_jax = np.load('eigvals_model_dpy_jax.npy')
np.testing.assert_array_almost_equal(DPT_eigvals_from_qdpy,
                                     eigvals_from_dpy_jax)
