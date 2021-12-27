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
from qdpy_jax import build_cenmult_and_nbs as build_cnm
from vorontsov_qdpy import sparse_precompute as precompute
from vorontsov_qdpy import sparse_precompute_bkm as precompute_bkm
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

#------------------------------------ precomputing -------------------------#
# precomputing the V11 bkm components
# all = here all the ctrl points above rth are present, later some of them are fixed
noc_bkm_all_sparse, fixed_bkm_all_sparse, k_arr, p_arr =\
                                precompute_bkm.build_bkm_all_cenmults()

# precomputing the supermatrix components
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr =\
                                precompute.build_hypmat_all_cenmults()

#-----------------------initializing shape parameters-----------------------#
nc = GVARS.nc
len_s = len(GVARS.s_arr)
len_s_fit = len(np.arange(smin_ind, smax_ind+1))
nmults = len(GVARS.n0_arr)
fixmat_shape = fixed_hypmat_all_sparse[0].shape
max_nbs = fixmat_shape[1]
len_mmax = fixmat_shape[2]
len_data = len(omega0_arr)
num_k = int((np.unique(k_arr)>0).sum())

#-----------------------------------------------------------------#
# flattening exact qdPy hypmat to facilitate densification
'''
fixed_hypmat = np.reshape(fixed_hypmat_all_sparse,
                          (nmults, max_nbs*max_nbs*len_mmax),
                          order='F')
noc_hypmat = np.reshape(noc_hypmat_all_sparse,
                        (nmults, len_s, nc, max_nbs*max_nbs*len_mmax),
                        order='F')

# flattened indices in sparse form for densification
sparse_idxs_flat = np.zeros((nmults, max_nbs*max_nbs*len_mmax, 2), dtype=int)

for i in range(nmults):
    sparse_idxs_flat[i] = np.reshape(sparse_idx[i],
                                     (max_nbs*max_nbs*len_mmax, 2),
                                     order='F')

# densifying qdPy fixed part to get dim_hyper
fixed_hypmat_dense = sparse.coo_matrix((fixed_hypmat[0],
                                        (sparse_idxs_flat[0, ..., 0],
                                        sparse_idxs_flat[0, ..., 1]))).toarray()
dim_hyper = fixed_hypmat_dense.shape[0]
'''
#------------------------------------------------------------------#
c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in range(smin_ind, smax_ind+1):
    c_fixed[sind, cind_arr] = 0.0
        
#------------------------------------------------------------------#
# this is the fixed part of the supermatrix
fixed_hypmat_sparse = np.zeros((nmults, max_nbs, max_nbs, len_mmax))
fixed_bkm_sparse = np.zeros((nmults, num_k, len_mmax))

for i in range(nmults):
    fixed_hypmat_sparse[i, ...] =\
                build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                 fixed_hypmat_all_sparse[i],
                                                 c_fixed, nc, len_s)
    fixed_bkm_sparse[i, ...] =\
                build_hm_sparse.build_hypmat_w_c(noc_bkm_all_sparse[i],
                                                 fixed_bkm_all_sparse[i],
                                                 c_fixed, nc, len_s)

#------------------------------------------------------------------#
# moving axis to facilitate easy dot product later
param_coeff = np.moveaxis(noc_hypmat_all_sparse, 0, 2)
param_coeff_bkm = np.moveaxis(noc_bkm_all_sparse, 0, 2)

# retaining the appropriate s indices
param_coeff = param_coeff[smin_ind: smax_ind + 1]
param_coeff_bkm = param_coeff_bkm[smin_ind: smax_ind + 1]

# retaining the appropriate c indices
param_coeff = param_coeff[:, cind_arr]
param_coeff_bkm = param_coeff_bkm[:, cind_arr]

#------------------------intermediate bkm test---------------------#                          
#----------do this only for n=0, l between 194 and 208-------------#                       
# constructing bkm full                                                                      
bkm_jax = np.sum(param_coeff_bkm * true_params[:, :, NAX, NAX, NAX], axis=(0,1)) \
          + fixed_bkm_sparse                 
dom_dell = GVARS.dom_dell                                                                     
bkm_scaled = -1.0 * bkm_jax / dom_dell[:, NAX, NAX]                                           

# loading precomputed benchmarked value                                                    
bkm_test = np.load('../tests/bkm_test.npy')                                               

# testing against a benchmarked values stored                                                 
np.testing.assert_array_almost_equal(bkm_scaled, bkm_test)                                  

#-----------generating the p * domega/dell factor----------------#                           
freq_diag = np.zeros_like(fixed_hypmat_sparse)

for i in range(nmults):
    CNM_AND_NBS = build_cnm.getnt4cenmult(GVARS.n0_arr[i],GVARS.ell0_arr[i],GVARS)
    omega0 = CNM_AND_NBS.omega_nbs[0]
    for j in range(max_nbs):
        freq_diag[i,j,j,:] =\
                (CNM_AND_NBS.omega_nbs[j]**2 - omega0**2)/(2*omega0)

#-----------------------------------------------------------------# 

# saving the supermatrix components
# np.savetxt('.dimhyper', np.array([dim_hyper]), fmt='%d')
np.save('fixed_part.npy', fixed_hypmat_sparse)
np.save('param_coeff.npy', param_coeff)
# np.save('sparse_idx.npy', sparse_idx)
np.save('omega0_arr.npy', omega0_arr)
np.save('dom_dell_arr.npy', GVARS.dom_dell)
np.save('freq_diag.npy', freq_diag)
np.save('ell0_arr.npy', ell0_arr)
np.save('data_model.npy', eigvals_model)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)
np.save('true_params.npy', true_params)

# saving the V11 bkm components
np.save('noc_bkm.npy', param_coeff_bkm)
np.save('fixed_bkm.npy', fixed_bkm_sparse)
np.save('k_arr.npy', k_arr)
np.save('p_arr.npy', p_arr)


# temporary test of clp
def get_clp(bkm):
    k_arr_denom = k_arr * 1
    k_arr_denom[k_arr==0] = np.inf

    tvals = np.linspace(0, jnp.pi, 25)

    # integrand of shape (ell, p, m ,t)
    integrand = np.zeros((p_arr.shape[0],
                          p_arr.shape[1],
                          p_arr.shape[2],
                          len(tvals)))

    for i in range(len(tvals)):
        term2 = 2*bkm*np.sin(k_arr*tvals[i])/k_arr_denom
        term2 = term2.sum(axis=1)
        integrand[:,:,:,i] = np.cos(p_arr*tvals[i] - term2[:, NAX, :])

    integral = np.trapz(integrand, axis=-1, x=tvals)/np.pi
    return integral

clp = get_clp(bkm_scaled)
