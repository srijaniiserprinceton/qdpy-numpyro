import os
#----------------setting the number of chains to be used-----------------#
# num_chains = 4
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
#------------------------------------------------------------------------# 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

import jax
from jax import random
from jax import jit
from jax import jacfwd, jacrev
from jax.lax import fori_loop as foril
from jax.lax import dynamic_slice as jdc
from jax.lax import dynamic_update_slice as jdc_update
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
from jax.experimental import sparse
print(jax.devices())

from dpy_jax import jax_functions_dpy as jf

from qdpy_jax import globalvars as gvar_jax
#------------------------------------------------------------------------#                    

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

#-------------loading precomputed files for the problem-------------------#                  
data = np.load('data_model.npy')
true_params_flat = np.load('true_params_flat.npy')
param_coeff_flat = np.load('param_coeff_flat.npy')
fixed_part = np.load('fixed_part.npy')
sparse_idx = np.load('sparse_idx.npy')
acoeffs_sigma_HMI = np.load('acoeffs_sigma_HMI.npy')
acoeffs_HMI = np.load('acoeffs_HMI.npy')
cind_arr = np.load('cind_arr.npy')
sind_arr = np.load('sind_arr.npy')
ell0_arr = np.load('ell0_arr.npy')
omega0_arr = np.load('omega0_arr.npy')
# Reading RL poly from precomputed file                                                      
# shape (nmults x (smax+1) x 2*ellmax+1)                                                     
RL_poly = np.load('RL_poly.npy')
sigma2scale = np.load('../dpy_jax/sigma2scale.npy')
#------------------------------------------------------------------------#

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)
dim_hyper = int(np.loadtxt('.dimhyper'))
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)

# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
# reshaping to (nmults x (smax+1) x dim_hyper)
Pjl = RL_poly[:, smin:smax+1:2, :]
'''
Pjl = np.zeros((Pjl_read.shape[0],
                Pjl_read.shape[1],
                dim_hyper))
Pjl[:, :, :Pjl_read.shape[2]] = Pjl_read
'''

#------------------------------------------------------------------------#                   
# calculating the denominator of a-coefficient converion apriori                             
# shape (nmults, num_j)                                                                      
aconv_denom = np.zeros((nmults, Pjl.shape[1]))
for mult_ind in range(nmults):
    aconv_denom[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

#-------------------------converting to device array---------------------#
Pjl = jnp.asarray(Pjl)
data = jnp.asarray(data)
true_params_flat = jnp.asarray(true_params_flat)
param_coeff_flat = jnp.asarray(param_coeff_flat)
fixed_part = jnp.asarray(fixed_part)
acoeffs_sigma_HMI = jnp.asarray(acoeffs_sigma_HMI)
sparse_idx = jnp.asarray(sparse_idx)
ell0_arr = jnp.asarray(ell0_arr)
omega0_arr = jnp.asarray(omega0_arr)
aconv_denom = jnp.asarray(aconv_denom)

#----------------------making the data_acoeffs---------------------------#
data_acoeffs = jnp.zeros(num_j*nmults)

def loop_in_mults(mult_ind, data_acoeff):
    ell0 = ell0_arr[mult_ind]
    data_omega = jdc(data, (mult_ind*dim_hyper,), (dim_hyper,))
    Pjl_local = Pjl[mult_ind]
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_local @ data_omega)/Pjl_norm[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

sys.exit()

def model(c_arr):
    pred_acoeffs = jnp.zeros(num_j * nmults)

    c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    pred = c_arr_denorm @ param_coeff + fixed_part

    def loop_in_mults(mult_ind, pred_acoeff):
        _eigval_mult = jnp.zeros(2*ellmax+1)
        ell0 = ell0_arr[mult_ind]
        omegaref = omega0_arr[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind], sparse_idx[mult_ind],
                                         shape=(dim_hyper, dim_hyper))
        _eigval_mult = get_eigs(pred_dense)[:2*ellmax+1]/2./omegaref*GVARS.OM*1e6

        Pjl_local = Pjl[mult_ind]
        
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

    misfit_acoeffs = (pred_acoeffs - data_acoeffs)/acoeffs_sigma

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





