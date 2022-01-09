import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mu", help="regularization",
                    type=float, default=0.)
PARGS = parser.parse_args()
#----------------setting the number of chains to be used-----------------#                    
# num_chains = 3
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
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

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.append(f"{package_dir}/plotter")
import postplotter
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
# model_params_sigma = np.load('model_params_sigma.npy')*100.
sigma2scale = np.load('sigma2scale.npy')

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
print(f"data_acoeffs = {data_acoeffs[:15]}")

# plotting acoeffs pred and data to see if we should expect got fit
pred_acoeffs_plot = np.reshape(pred_acoeffs, (3,-1), 'F')
data_acoeffs_plot = np.reshape(data_acoeffs, (3,-1), 'F')

for i in range(3):
    plt.figure()
    plt.plot(pred_acoeffs_plot[i], '.r', markersize=2)
    plt.plot(data_acoeffs_plot[i], '.k', markersize=2)
    plt.savefig(f'a{2*i+1}.png')
    plt.close()

#----------------------------------------------------------------------# 
# the regularizing parameter
mu = PARGS.mu

# the length of data
len_data = len(data_acoeffs)

# the model function that is used by MCMC kernel
def data_misfit_fn(c_arr):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)
            
    # denormalizing to make actual model params
    c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    pred = fixed_part + c_arr_denorm @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    data_misfit_arr = (pred_acoeffs - data_acoeffs)/acoeffs_sigma_HMI

    return jnp.sum(jnp.square(data_misfit_arr))

def data_misfit_arr_fn(c_arr):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)
            
    # denormalizing to make actual model params
    c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    pred = fixed_part + c_arr_denorm @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    data_misfit_arr = (pred_acoeffs - data_acoeffs)/acoeffs_sigma_HMI

    return data_misfit_arr



def model_misfit_fn(c_arr):
    return jnp.sum(jnp.square(c_arr))

def hessian(f):
    return jacfwd(jacrev(f))

data_hess_fn = hessian(data_misfit_fn)

def loss_fn(c_arr):
    data_misfit_val = data_misfit_fn(c_arr)
    model_misfit_val = model_misfit_fn(c_arr)
    data_hess = data_hess_fn(c_arr)
    lambda_factor = jnp.trace(data_hess) / len_data

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


#---------------------- jitting the functions --------------------------#
_grad_fn = jit(grad_fn)
_hess_fn = jit(hess_fn)
_update_H = jit(update_H)
_loss_fn = jit(loss_fn)

#-----------------------the main training loop--------------------------#
# initialization of params
c_init = np.random.uniform(5.0, 20.0, size=len(true_params_flat))

#------------------plotting the initial profiles-------------------#                     
c_arr_init_full = jf.c4fit_2_c4plot(GVARS, c_init*true_params_flat,
                                    sind_arr, cind_arr)

# converting ctrl points to wsr and plotting                                                  
init_plot = postplotter.postplotter(GVARS, c_arr_init_full, 'init')
#------------------------------------------------------------------------# 

# getting the renormalized model parameters
c_arr_renorm = jf.model_renorm(c_init*true_params_flat,
                               true_params_flat,
                               sigma2scale)
c_arr_renorm = jnp.asarray(c_arr_renorm)

N = len(data_acoeffs)

loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-10
maxiter = 15
itercount = 0

t1s = time.time()
while ((abs(loss_diff) > loss_threshold) and
       (itercount < maxiter)):
    t1 = time.time()
    loss_prev = loss
    grads = _grad_fn(c_arr_renorm)
    hess = _hess_fn(c_arr_renorm)
    hess_inv = jnp.linalg.inv(hess)
    c_arr_renorm = _update_H(c_arr_renorm, grads, hess_inv)
    loss = _loss_fn(c_arr_renorm)

    model_misfit = model_misfit_fn(c_arr_renorm)
    data_hess = data_hess_fn(c_arr_renorm)
    model_misfit = model_misfit * jnp.trace(data_hess) / len_data
    data_misfit = loss - model_misfit * mu

    loss_diff = loss_prev - loss
    loss_arr.append(loss)
    itercount += 1
    t2 = time.time()
    print(f'[{itercount:3d} | {(t2-t1):6.1f} sec ] data_misfit = {data_misfit:12.5e} loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {abs(grads).max():12.5e} model_misfit={model_misfit:12.5e}')


    t2 = time.time()

t2s = time.time()
print(f"Total time taken = {(t2s-t1s):12.3f} seconds")

# reconverting back to model_params in units of true_params_flat
c_arr_fit = jf.model_denorm(c_arr_renorm, true_params_flat, sigma2scale)\
            /true_params_flat

print(c_arr_fit)

#------------------plotting the post fitting profiles-------------------#
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                   sind_arr, cind_arr)

# converting ctrl points to wsr and plotting                                                  
fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full, 'fit')

#------------------------------------------------------------------------#

with open("reg_misfit.txt", "a") as f:
    f.seek(0, os.SEEK_END)
    opstr = f"{mu:18.12e}, {data_misfit:18.12e}, {model_misfit:18.12e}\n"
    f.write(opstr)

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

