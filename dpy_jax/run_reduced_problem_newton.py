import os

#----------------setting the number of chains to be used-----------------#                    
num_chains = 3
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
#------------------------------------------------------------------------# 

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import arviz as az
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
print(jax.devices())

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA

from qdpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions_dpy as jf
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
acoeffs_sigma_HMI = np.load('acoeffs_sigma_HMI.npy')
acoeffs_HMI = np.load('acoeffs_HMI.npy')
cind_arr = np.load('cind_arr.npy')
sind_arr = np.load('sind_arr.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load('RL_poly.npy')
model_params_sigma = np.load('model_params_sigma.npy')

#------------------------------------------------------------------------# 

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)
dim_hyper = 2 * np.max(GVARS.ell0_arr) + 1
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
# number of s to fit
len_s = len(sind_arr)
# number of c's to fit
nc = len(cind_arr)

# slicing the Pjl correctly in angular degree s
Pjl = RL_poly[:, smin:smax+1:2, :]

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
acoeffs_HMI = jnp.asarray(acoeffs_HMI)
acoeffs_sigma_HMI = jnp.asarray(acoeffs_sigma_HMI)
aconv_denom = jnp.asarray(aconv_denom)

#----------------------making the data_acoeffs---------------------------# 
data_acoeffs = jnp.zeros(num_j*nmults)

def loop_in_mults(mult_ind, data_acoeff):
    data_omega = jdc(data, (mult_ind*dim_hyper,), (dim_hyper,))
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl[mult_ind] @ data_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

#---------------checking that the loaded data are correct----------------#
pred = fixed_part * 1.0

# adding the contribution from the fitting part
pred += true_params_flat @ param_coeff_flat

# these arrays should be very close
np.testing.assert_array_almost_equal(pred, data)

#-------------checking that the acoeffs match correctly------------------#
pred_acoeffs = jnp.zeros(num_j * nmults)

def loop_in_mults(mult_ind, pred_acoeff):
    pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
    pred_acoeff = jdc_update(pred_acoeff,
                             (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))

    return pred_acoeff

pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)

#----------------------------------------------------------------------#
# changing to the HMI acoeffs if doing this for real data 
# data_acoeffs = GVARS.acoeffs_true

# the regularizing parameter
mu = 1.e-3

# the model function that is used by MCMC kernel
def data_misfit_fn(c_arr):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)
                          
    c_arr = c_arr * true_params_flat

    pred = fixed_part + c_arr @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    data_misfit_arr = (pred_acoeffs - data_acoeffs)/acoeffs_sigma_HMI

    return jnp.sum(jnp.square(data_misfit_arr))

def model_misfit_fn(c_arr):
    # getting the renormalized model parameters
    model_misfit_arr = jf.model_renorm(c_arr*true_params_flat, true_params_flat, model_params_sigma)
    return jnp.sum(jnp.square(model_misfit_arr))

def hessian(f):
    return jacfwd(jacrev(f))

data_hess_fn = hessian(data_misfit_fn)

def loss_fn(c_arr):
    data_misfit_val = data_misfit_fn(c_arr)
    model_misfit_val = model_misfit_fn(c_arr)
    data_hess = data_hess_fn(c_arr)
    lambda_factor = jnp.trace(data_hess)
    # total misfit
    misfit = data_misfit_val + mu * model_misfit_val * lambda_factor
    return misfit

grad_fn = jax.grad(loss_fn)
hess_fn = hessian(loss_fn)

def update(c_arr, grads, loss):
    grad_strength = jnp.sqrt(jnp.sum(jnp.square(grads)))
    return jax.tree_multimap(lambda c, g: c - g / grad_strength, c_arr, grads)

def update_H(c_arr, grads, hess_inv):
    # grad_strength = jnp.sqrt(jnp.sum(jnp.square(grads)))
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)


#-----------------------the main training loop--------------------------#
# initialization of params
c_arr = np.random.uniform(5.0, 20.0, size=len(true_params_flat))

N = len(data_acoeffs)

loss = 1e25
loss_arr = []
loss_threshold = 1e-12

while (loss > loss_threshold):
    grads = grad_fn(c_arr)
    hess = hess_fn(c_arr)
    hess_inv = jnp.linalg.inv(hess)
    c_arr = update_H(c_arr, grads, hess_inv)
    loss = loss_fn(c_arr)
    loss_arr.append(loss)
    print(f'Loss = {loss:12.5e}; max-grads = {abs(grads).max():12.5e}')

#------------------------------------------------------------------------# 
def print_summary(samples, ctrl_arr):
    keys = samples.keys()
    for key in keys:
        sample = samples[key]
        key_split = key.split("_")
        idx = int(key_split[-1])
        sidx = int((int(key_split[0][1])-1)//2)
        obs = ctrl_arr[sidx-1, idx] / 1e-3
        print(f"[{obs:11.4e}] {key}: {sample.mean():.4e} +/- {sample.std():.4e}:" +
              f"error/sigma = {(sample.mean()-obs)/sample.std():8.3f}")
    return None
#------------------------------------------------------------------------# 

