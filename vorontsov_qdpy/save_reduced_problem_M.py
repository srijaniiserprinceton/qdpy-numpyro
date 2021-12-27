import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse
from jax.experimental import sparse as jsparse

import jax
from jax import random
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.lax import fori_loop as foril
from jax.ops import index_update as jidx_update
from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)

config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

# new package in jax.numpy
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_cenmult_and_nbs as build_cnm
from vorontsov_qdpy import sparse_precompute_M as precompute_M
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

NAX = np.newaxis

#-------------------------------------------------------------#
# the indices of ctrl points that we want to investigate
ind_min, ind_max = 0, 1
cind_arr = np.arange(ind_min, ind_max + 1)

# the angular degree s that we want to investigate 
smin, smax = 3, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.arange(smin_ind, smax_ind+1)

#-------------------------------------------------------------#

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))
GVARS_PATHS, GVARS_TR, __ = GVARS.get_all_GVAR()

eigvals_model = np.load("evals_model.npy")
eigvals_model = jnp.asarray(eigvals_model)
eigvals_sigma = jnp.asarray(np.load('eigvals_sigma.npy'))
acoeffs_sigma = jnp.asarray(np.load('acoeffs_sigma.npy'))
num_eigvals = len(eigvals_model)

#-------------------------generating the true parameters-------------------#
true_params = np.zeros((smax_ind - smin_ind + 1,
                        ind_max - ind_min + 1))

true_params = GVARS.ctrl_arr_dpt_clipped[sind_arr][:, cind_arr]


# precomputing the M supermatrix components
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr =\
                                precompute_M.build_hypmat_all_cenmults()
noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)

#-----------------------initializing shape parameters-------------------------#                
nc = GVARS.nc
len_s = len(GVARS.s_arr)
len_s_fit = len(np.arange(smin_ind, smax_ind+1))
nmults = len(GVARS.n0_arr)
fixmat_shape = fixed_hypmat_all_sparse[0].shape
max_nbs = fixmat_shape[1]
len_mmax = fixmat_shape[2]
len_data = len(omega0_arr)

#-----------------------------------------------------------------------------#
fixed_hypmat = np.reshape(fixed_hypmat_all_sparse,
                          (nmults, max_nbs*max_nbs*len_mmax),
                          order='F')
noc_hypmat = np.reshape(noc_hypmat_all_sparse,
                        (nmults, len_s, nc, max_nbs*max_nbs*len_mmax),
                        order='F')

#----------------------------------------------------------------------------# 
c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in sind_arr:
    c_fixed[sind, cind_arr] = 0.0

# this is the fixed part of the hypermatrix
fixed_hypmat_sparse = np.zeros((nmults, max_nbs, max_nbs, len_mmax))

for i in range(nmults):
    fixed_hypmat_sparse[i, ...] =\
                build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                 fixed_hypmat_all_sparse[i],
                                                 c_fixed, nc, len_s)

#------------------------------------------------------------------#
# moving axis to facilitate easy dot product later                                           
param_coeff = np.moveaxis(noc_hypmat_all_sparse, 0, 2)

# retaining the appropriate s indices                                                       
param_coeff = param_coeff[smin_ind: smax_ind + 1]

# retaining the appropriate c indices                                                       
param_coeff = param_coeff[:, cind_arr]

#-----------generating the p * domega/dell factor----------------#
p_dom_dell = np.zeros_like(fixed_hypmat_sparse)

for i in range(nmults):
    CNM_AND_NBS = build_cnm.getnt4cenmult(GVARS.n0_arr[i],GVARS.ell0_arr[i],GVARS)
    ell0 = CNM_AND_NBS.nl_nbs[0,1]
    for j in range(max_nbs):
        p_dom_dell[i,j,j,:] = CNM_AND_NBS.nl_nbs[j,1] - ell0

# scaling with inverse of domega/dell
p_dom_dell *= GVARS.dom_dell[:,NAX,NAX,NAX]

#---------------saving the supermatrix components-----------------#
print(f"param_coeff = {param_coeff.shape}")
print(f"cind_arr = {cind_arr}")
np.save('fixed_part_M.npy', fixed_hypmat_sparse)
np.save('param_coeff_M.npy', param_coeff)
np.save('p_dom_dell.npy', p_dom_dell)
sys.exit()

#--------------------testing M with qdPy submatrices------------------------#

fixed_hypmat_dense = sparse.coo_matrix((fixed_hypmat[0],
                                        (sparse_idxs_flat[0, ..., 0],
                                        sparse_idxs_flat[0, ..., 1]))).toarray()

#-----------------------------------------------------------------# 


