B'''
import os
num_chains = 38
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
'''
import numpy as np
import jax
from jax import random
from jax import jit
from jax import jacfwd, jacrev
from jax.lax import fori_loop as foril
from jax.lax import dynamic_slice as jdc
from jax.lax import dynamic_update_slice as jdc_update
from jax.ops import index as jidx
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
jidx_update = jax.ops.index_update
import sys
from dpy_jax import globalvars as gvar_jax

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)

dim_hyper = 2 * np.max(GVARS.ell0_arr) + 1

# loading the files forthe problem
data = np.load('data_model.npy')
# true params from Antia wsr
true_params = np.load('true_params.npy')
param_coeff = np.load('param_coeff.npy')
acoeffs_sigma = np.load('acoeffs_sigma.npy')
fixed_part = np.load('fixed_part.npy')
cind_arr = np.load('cind_arr.npy')
smin_ind, smax_ind = np.load('sind_arr.npy')

# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load('RL_poly.npy')
# picking out only the odd s
# smin, smax = 2*smin_ind+1, 2*smax_ind+1
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
Pjl = RL_poly[:, smin:smax+1:2, :]

# calculating the denominator of a-coefficient converion apriori
# shape (nmults, num_j)
aconv_denom = np.zeros((nmults, Pjl.shape[1]))
for mult_ind in range(nmults):
    aconv_denom[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

# number of s to fit
len_s = true_params.shape[0]
# number of c's to fit
nc = true_params.shape[1]

# number of data points 
len_data = len(data)

# converting to device array
Pjl = jnp.asarray(Pjl)
data = jnp.asarray(data)
true_params = jnp.asarray(true_params)
param_coeff = jnp.asarray(param_coeff)
fixed_part = jnp.asarray(fixed_part)
acoeffs_sigma = jnp.asarray(acoeffs_sigma)
aconv_denom = jnp.asarray(aconv_denom)

# making the data_acoeffs
data_acoeffs = jnp.zeros(num_j*nmults)

def loop_in_mults(mult_ind, data_acoeff):
    data_omega = jdc(data, (mult_ind*dim_hyper,), (dim_hyper,))
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl[mult_ind] @ data_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

######################################################
# checking that the loaded data are correct
pred = fixed_part * 1.0

# adding the contribution from the fitting part
for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        pred += true_params[sind-1, ci] * param_coeff[sind-1][ci]

# these arrays should be very close
np.testing.assert_array_almost_equal(pred, data)

######################################################
# checking that the loaded data are correct                                                   
pred_acoeffs = jnp.zeros(num_j * nmults)

pred = fixed_part * 1.0

# adding the contribution from the fitting part
for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        pred += true_params[sind-1, ci] * param_coeff[sind-1][ci]

def loop_in_mults(mult_ind, pred_acoeff):
    pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
    pred_acoeff = jdc_update(pred_acoeff,
                             (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    return pred_acoeff

pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)

######################################################

# num_params = len(cind_arr)

# flattening out the (s, nc) dimensions to easy dot product
true_params = np.reshape(true_params, (nc * len_s))
param_coeff = np.reshape(param_coeff, (nc * len_s, -1))


def loss_fn(c_arr):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)

    # predicted frequency shifts
    pred = fixed_part + c_arr @ param_coeff
    
    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    misfit_acoeffs = (pred_acoeffs - data_acoeffs)/acoeffs_sigma

    return jnp.mean(jnp.square(misfit_acoeffs))

# function returning the hessian inverse
def hessian(f):
    return jacfwd(jacrev(f))

# using autograd
grad_fn = jax.grad(loss_fn)
hess_fn = hessian(loss_fn)

def update(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)

# initial c_arr
c_init = np.random.uniform(0.01*true_params, 2.0*true_params, size= nc*len_s)

# the array to interate over
c_arr = c_init.copy()


# max number of iterations
Niter = 1000
# initializing to some positive number
loss = 10

def cond_fun(func_args):
    __, loss = func_args
    return loss > 1e-10

# iterating to solution
def loop_iter(func_args):
    c_arr, loss = func_args
    # computing the misfit
    loss = loss_fn(c_arr)
    # computing the gradient
    grads = grad_fn(c_arr)
    # computing the hessian
    hess = hess_fn(c_arr)
    # computing the inverse of the hessian
    hess_inv = jnp.linalg.inv(hess)
    # stepping in ctrl array space
    c_arr = update(c_arr, grads, hess_inv)

    return c_arr, loss

# jitting the loop functions
loop_iter_ = jax.jit(loop_iter)    

# while loop
c_arr, loss = jax.lax.while_loop(cond_fun, loop_iter_, (c_arr, loss))


def print_summary(inv_params, true_params):
    for i in range(len(true_params)):
        obs = true_params[i]
        inv = inv_params[i]
        print(f"[{obs:11.4e}] {inv:.4e}")
    return None

print_summary(c_arr, true_params)
