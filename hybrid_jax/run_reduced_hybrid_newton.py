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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from jax.experimental import sparse
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
print(jax.devices())

from qdpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions_dpy as jf

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.append(f"{package_dir}/plotter")
import postplotter
import plot_acoeffs_datavsmodel as plot_acoeffs
#------------------------------------------------------------------------# 
# defining the directories for dpy_jax and qdpy_jax
dpy_dir = "../dpy_jax"
qdpy_dir = "../qdpy_jax"

#------------------------------------------------------------------------# 
#                                DPY-JAX                                 #
#------------------------------------------------------------------------#
ARGS_D = np.loadtxt(f"{dpy_dir}/.n0-lmin-lmax.dat")
GVARS_D = gvar_jax.GlobalVars(n0=int(ARGS_D[0]),
                              lmin=int(ARGS_D[1]),
                              lmax=int(ARGS_D[2]),
                              rth=ARGS_D[3],
                              knot_num=int(ARGS_D[4]),
                              load_from_file=int(ARGS_D[5]),
                              relpath=dpy_dir)

#-------------loading precomputed files for the problem-------------------# 
data_D = np.load(f'{dpy_dir}/data_model.npy')
true_params_flat_D = np.load(f'{dpy_dir}/true_params_flat.npy')
param_coeff_flat_D = np.load(f'{dpy_dir}/param_coeff_flat.npy')
fixed_part_D = np.load(f'{dpy_dir}/fixed_part.npy')
acoeffs_sigma_HMI_D = np.load(f'{dpy_dir}/acoeffs_sigma_HMI.npy')
acoeffs_HMI_D = np.load(f'{dpy_dir}/acoeffs_HMI.npy')
cind_arr_D = np.load(f'{dpy_dir}/cind_arr.npy')
sind_arr_D = np.load(f'{dpy_dir}/sind_arr.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly_D = np.load(f'{dpy_dir}/RL_poly.npy')
sigma2scale = np.load(f'{dpy_dir}/sigma2scale.npy')
D_bsp_j_D_bsp_k = np.load(f'{dpy_dir}/D_bsp_j_D_bsp_k.npy')
#------------------------------------------------------------------------# 

nmults_D = len(GVARS_D.ell0_arr)
num_j_D = len(GVARS_D.s_arr)
dim_hyper_D = 2 * np.max(GVARS_D.ell0_arr) + 1
smin_D = min(GVARS_D.s_arr)
smax_D = max(GVARS_D.s_arr)
# number of s to fit
len_s_D = len(sind_arr_D)
# number of c's to fit
nc_D = len(cind_arr_D)

# slicing the Pjl correctly in angular degree s
Pjl_D = RL_poly_D[:, smin_D:smax_D+1:2, :]

#------------------------------------------------------------------------#
# calculating the denominator of a-coefficient converion apriori
# shape (nmults, num_j)
aconv_denom_D = np.zeros((nmults_D, Pjl_D.shape[1]))
for mult_ind in range(nmults_D):
    aconv_denom_D[mult_ind] = np.diag(Pjl_D[mult_ind] @ Pjl_D[mult_ind].T)

#-------------------------converting to device array---------------------# 
Pjl_D = jnp.asarray(Pjl_D)
data_D = jnp.asarray(data_D)
true_params_flat_D = jnp.asarray(true_params_flat_D)
param_coeff_flat_D = jnp.asarray(param_coeff_flat_D)
fixed_part_D = jnp.asarray(fixed_part_D)
acoeffs_HMI_D = jnp.asarray(acoeffs_HMI_D)
acoeffs_sigma_HMI_D = jnp.asarray(acoeffs_sigma_HMI_D)
aconv_denom_D = jnp.asarray(aconv_denom_D)

#----------------------making the data_acoeffs---------------------------# 
data_acoeffs_D = jnp.zeros(num_j_D*nmults_D)

def loop_in_mults_D(mult_ind, data_acoeff):
    data_omega = jdc(data_D, (mult_ind*dim_hyper_D,), (dim_hyper_D,))
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_D[mult_ind] @ data_omega)/aconv_denom_D[mult_ind],
                             (mult_ind * num_j_D,))
    
    return data_acoeff

data_acoeffs_D = foril(0, nmults_D, loop_in_mults_D, data_acoeffs_D)

#---------------checking that the loaded data are correct----------------#
pred_D = fixed_part_D * 1.0

# adding the contribution from the fitting part
pred_D += true_params_flat_D @ param_coeff_flat_D

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_D, data_D)

#-------------checking that the acoeffs match correctly------------------#
pred_acoeffs_D = jnp.zeros(num_j_D * nmults_D)

def loop_in_mults_D(mult_ind, pred_acoeff):
    pred_omega = jdc(pred_D, (mult_ind*dim_hyper_D,), (dim_hyper_D,))
    pred_acoeff = jdc_update(pred_acoeff,
                             (Pjl_D[mult_ind] @ pred_omega)/aconv_denom_D[mult_ind],
                             (mult_ind * num_j_D,))

    return pred_acoeff

pred_acoeffs_D = foril(0, nmults_D, loop_in_mults_D, pred_acoeffs_D)

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs_D, data_acoeffs_D)

#----------------------------------------------------------------------#
# changing to the HMI acoeffs if doing this for real data 
# data_acoeffs_D = GVARS_D.acoeffs_true
np.random.seed(3)
data_acoeffs_err_D = np.random.normal(loc=0, scale=acoeffs_sigma_HMI_D)
data_acoeffs_D = data_acoeffs_D + data_acoeffs_err_D
data_acoeffs_out_HMI_D = GVARS_D.acoeffs_out_HMI
print(f"data_acoeffs_D = {data_acoeffs_D[:15]}")

#----------------------------------------------------------------------# 
# plotting acoeffs pred and data to see if we should expect good fit
plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs_D, data_acoeffs_D,
                                      data_acoeffs_out_HMI_D,
                                      acoeffs_sigma_HMI_D, 'ref_D')

#------------------------------------------------------------------------#                    
#                                QDPY-JAX                                #                   
#------------------------------------------------------------------------# 
ARGS_Q = np.loadtxt(f"{qdpy_dir}/.n0-lmin-lmax.dat")
GVARS_Q = gvar_jax.GlobalVars(n0=int(ARGS_Q[0]),
                              lmin=int(ARGS_Q[1]),
                              lmax=int(ARGS_Q[2]),
                              rth=ARGS_Q[3],
                              knot_num=int(ARGS_Q[4]),
                              load_from_file=int(ARGS_Q[5]),
                              relpath=qdpy_dir)
#-------------loading precomputed files for the problem-------------------#                  
data_Q = np.load(f'{qdpy_dir}/data_model.npy')
true_params_flat_Q = np.load(f'{qdpy_dir}/true_params_flat.npy')
param_coeff_flat_Q = np.load(f'{qdpy_dir}/param_coeff_flat.npy')
fixed_part_Q = np.load(f'{qdpy_dir}/fixed_part.npy')
sparse_idx_Q = np.load(f'{qdpy_dir}/sparse_idx.npy')
acoeffs_sigma_HMI_Q = np.load(f'{qdpy_dir}/acoeffs_sigma_HMI.npy')
acoeffs_HMI_Q = np.load(f'{qdpy_dir}/acoeffs_HMI.npy')
cind_arr_Q = np.load(f'{qdpy_dir}/cind_arr.npy')
sind_arr_Q = np.load(f'{qdpy_dir}/sind_arr.npy')
ell0_arr_Q = np.load(f'{qdpy_dir}/ell0_arr.npy')
omega0_arr_Q = np.load(f'{qdpy_dir}/omega0_arr.npy')
# Reading RL poly from precomputed file                                                      
# shape (nmults x (smax+1) x 2*ellmax+1)                                                     
RL_poly_Q = np.load(f'{qdpy_dir}/RL_poly.npy')

#-------------------Miscellaneous parameters---------------------------#                    
nmults_Q = len(GVARS_Q.ell0_arr)
num_j_Q = len(GVARS_Q.s_arr)
dim_hyper_Q = int(np.loadtxt(f'{qdpy_dir}/.dimhyper'))
ellmax_Q = np.max(ell0_arr_Q)
smin_Q = min(GVARS_Q.s_arr)
smax_Q = max(GVARS_Q.s_arr)
# Reading RL poly from precomputed file                                                      
# shape (nmults x (smax+1) x 2*ellmax+1)                                                     
Pjl_Q = RL_poly_Q[:, smin_Q:smax_Q+1:2, :]

#------------------------------------------------------------------------#                   
# calculating the denominator of a-coefficient converion apriori                             
# shape (nmults, num_j)                                                                      
aconv_denom_Q = np.zeros((nmults_Q, Pjl_Q.shape[1]))
for mult_ind in range(nmults_Q):
    aconv_denom_Q[mult_ind] = np.diag(Pjl_Q[mult_ind] @ Pjl_Q[mult_ind].T)

#-------------------------converting to device array---------------------#                   
Pjl_Q = jnp.asarray(Pjl_Q)
data_Q = jnp.asarray(data_Q)
true_params_flat_Q = jnp.asarray(true_params_flat_Q)
param_coeff_flat_Q = jnp.asarray(param_coeff_flat_Q)
fixed_part_Q = jnp.asarray(fixed_part_Q)
acoeffs_sigma_HMI_Q = jnp.asarray(acoeffs_sigma_HMI_Q)
sparse_idx_Q = jnp.asarray(sparse_idx_Q)
ell0_arr_Q = jnp.asarray(ell0_arr_Q)
omega0_arr_Q = jnp.asarray(omega0_arr_Q)
aconv_denom_Q = jnp.asarray(aconv_denom_Q)

#----------------------making the data_acoeffs---------------------------#                    
data_acoeffs_Q = jnp.zeros(num_j_Q*nmults_Q)

def loop_in_mults_Q(mult_ind, data_acoeff):
    data_omega = jdc(data_Q, (mult_ind*(2*ellmax_Q+1),), (2*ellmax_Q+1,))
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_Q[mult_ind] @ data_omega)/aconv_denom_Q[mult_ind],
                             (mult_ind * num_j_Q,))

    return data_acoeff

data_acoeffs_Q = foril(0, nmults_Q, loop_in_mults_Q, data_acoeffs_Q)

#-------------------functions for the forward model------------------------#                  
def model_Q(c_arr):
    pred_acoeffs = jnp.zeros(num_j_Q * nmults_Q)

    c_arr_denorm = jf.model_denorm(c_arr, true_params_flat_Q, sigma2scale)
    pred = c_arr_denorm @ param_coeff_flat_Q + fixed_part_Q

    def loop_in_mults(mult_ind, pred_acoeff):
        _eigval_mult = jnp.zeros(2*ellmax_Q+1)
        ell0 = ell0_arr_Q[mult_ind]
        omegaref = omega0_arr_Q[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind], sparse_idx_Q[mult_ind],
                                         shape=(dim_hyper_Q, dim_hyper_Q))
        _eigval_mult = get_eigs(pred_dense)[:2*ellmax_Q+1]/2./omegaref*GVARS_Q.OM*1e6

        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_Q[mult_ind] @ _eigval_mult)/aconv_denom_Q[mult_ind],
                                 (mult_ind * num_j_Q,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults_Q, loop_in_mults, pred_acoeffs)

    return pred_acoeffs

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

#----------------checking if forward model works fine--------------------#                    
c_arr_renorm_Q = jf.model_renorm(true_params_flat_Q, true_params_flat_Q, sigma2scale)
c_arr_renorm_Q = jnp.asarray(c_arr_renorm_Q)

model_Q_ = jit(model_Q)
pred_acoeffs_Q = model_Q_(c_arr_renorm_Q)

#-------------checking that the acoeffs match correctly------------------#
np.testing.assert_array_almost_equal(pred_acoeffs_Q, data_acoeffs_Q)
sys.exit()
#------------------------------------------------------------------------# 

# the regularizing parameter
mu = PARGS.mu


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
    # c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    # Djk is the same for s=3 and s=5
    cd1 = c_arr[0::3]
    cd3 = c_arr[1::3]
    cd5 = c_arr[2::3]
    Djk = D_bsp_j_D_bsp_k[0::3, 0::3]
    return (cd1 @ Djk @ cd1 + 
            cd3 @ Djk @ cd3 +
            cd5 @ Djk @ cd5)


def hessian(f):
    return jacfwd(jacrev(f))

data_hess_fn = hessian(data_misfit_fn)

def loss_fn(c_arr):
    data_misfit_val = data_misfit_fn(c_arr)
    model_misfit_val = model_misfit_fn(c_arr)
    data_hess = data_hess_fn(c_arr)
    lambda_factor = jnp.trace(data_hess) / len_data

    # total misfit
    misfit = data_misfit_val + mu * model_misfit_val #* lambda_factor
    return misfit

grad_fn = jax.grad(loss_fn)
hess_fn = hessian(loss_fn)

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
# c_init = np.ones_like(true_params_flat)

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

#----------------------------------------------------------------------#                      
# plotting acoeffs from initial data and HMI data
init_acoeffs = data_misfit_arr_fn(c_arr_renorm)*acoeffs_sigma_HMI +\
               data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'init')
#----------------------------------------------------------------------#

loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-9
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
    # data_hess = data_hess_fn(c_arr_renorm)
    # model_misfit = model_misfit * jnp.trace(data_hess) / len_data
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


#----------------------------------------------------------------------#                      
# plotting acoeffs from initial data and HMI data                                            
final_acoeffs = data_misfit_arr_fn(c_arr_renorm)*acoeffs_sigma_HMI +\
                data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'final')
#----------------------------------------------------------------------# 

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

# plotting the hessians for analysis
fig, ax = plt.subplots(1, 2, figsize=(10,5))

im1 = ax[0].pcolormesh(hess)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax[1].pcolormesh(data_hess)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

plt.tight_layout()
plt.savefig('hessians.png')
plt.close()
