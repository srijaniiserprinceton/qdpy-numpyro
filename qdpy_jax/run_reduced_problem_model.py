import os
#----------------setting the number of chains to be used-----------------#
# num_chains = 4
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
#------------------------------------------------------------------------# 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import time

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
outdir = f"{GVARS.scratch_dir}/qdpy_jax"
dpy_outdir = f"{GVARS.scratch_dir}/dpy_jax"
#-------------loading precomputed files for the problem-------------------#
data = np.load(f'{outdir}/data_model.npy')
true_params_flat = np.load(f'{outdir}/true_params_flat.npy')
param_coeff_flat = np.load(f'{outdir}/param_coeff_flat.npy')
fixed_part = np.load(f'{outdir}/fixed_part.npy')
sparse_idx = np.load(f'{outdir}/sparse_idx.npy')
acoeffs_sigma_HMI = np.load(f'{outdir}/acoeffs_sigma_HMI.npy')
acoeffs_HMI = np.load(f'{outdir}/acoeffs_HMI.npy')
cind_arr = np.load(f'{outdir}/cind_arr.npy')
sind_arr = np.load(f'{outdir}/sind_arr.npy')
ell0_arr = np.load(f'{outdir}/ell0_arr.npy')
omega0_arr = np.load(f'{outdir}/omega0_arr.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
RL_poly = np.load(f'{outdir}/RL_poly.npy')
sigma2scale = np.load(f'{dpy_outdir}/sigma2scale.npy')

#-------------------Miscellaneous parameters---------------------------#
nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)
dim_hyper = int(np.loadtxt('.dimhyper'))
ellmax = np.max(ell0_arr)
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
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
acoeffs_sigma_HMI = jnp.asarray(acoeffs_sigma_HMI)
sparse_idx = jnp.asarray(sparse_idx)
ell0_arr = jnp.asarray(ell0_arr)
omega0_arr = jnp.asarray(omega0_arr)
aconv_denom = jnp.asarray(aconv_denom)

#-------------------functions for the forward model------------------------#
def model(c_arr):
    pred_acoeffs = jnp.zeros(num_j * nmults)

    c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    pred = c_arr_denorm @ param_coeff_flat + fixed_part

    def loop_in_mults(mult_ind, pred_acoeff):
        _eigval_mult = jnp.zeros(2*ellmax+1)
        ell0 = ell0_arr[mult_ind]
        omegaref = omega0_arr[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind], sparse_idx[mult_ind],
                                         shape=(dim_hyper, dim_hyper))
        _eigval_mult = get_eigs(pred_dense)[:2*ellmax+1]/2./omegaref*GVARS.OM*1e6

        Pjl_local = Pjl[mult_ind]
        
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/aconv_denom[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

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
c_arr_renorm = jf.model_renorm(true_params_flat, true_params_flat, sigma2scale)
c_arr_renorm = jnp.asarray(c_arr_renorm)

model_ = jit(model)
pred_acoeffs = model_(c_arr_renorm)

#----------------------making the data_acoeffs---------------------------#
data_acoeffs = jnp.zeros(num_j*nmults)

def loop_in_mults(mult_ind, data_acoeff):
    ell0 = ell0_arr[mult_ind]
    data_omega = jdc(data, (mult_ind*(2*ellmax+1),), (2*ellmax+1,))
    Pjl_local = Pjl[mult_ind]
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_local @ data_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

#----------------------testing the pred_acoeffs---------------------------#
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)


#--------------------------------inversion---------------------------------#
def data_misfit_fn(c_arr):
    pred_acoeffs = model(c_arr)
    data_misfit_arr = (pred_acoeffs - data_acoeffs)/acoeffs_sigma_HMI
    return jnp.sum(jnp.square(data_misfit_arr))


def hessian(f):
    return jacfwd(jacfwd(f))

def update_H(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)

def approx_B(B_k, y_k, dx_k):
    B_k_dx_k = B_k @ dx_k
    B_kp1 = B_k + jnp.outer(y_k, y_k) / (y_k @ dx_k) -\
            jnp.outer(B_k_dx_k, B_k_dx_k) / (dx_k @ B_k @ dx_k)
    return B_kp1

# jitting functions
data_hess_fn = hessian(data_misfit_fn)
_data_hess_fn = jit(data_hess_fn)
_grad_fn = jit(jax.grad(data_misfit_fn))
_loss_fn = jit(data_misfit_fn)
_update_H = jit(update_H)
_approx_B = jit(approx_B)

# initializing c_arr_renorm
c_arr_renorm = 1.1 * np.ones_like(true_params_flat)
len_c = len(c_arr_renorm)

# the current approximate hessian
B = _data_hess_fn(c_arr_renorm)
B_arr = np.zeros_like(B)
B_arr = np.reshape(B_arr, (1, len_c, len_c))
B_arr[0] = B + np.identity(len_c) * 0.1

# true_hessian array over iteration
H_arr = np.zeros_like(B_arr)
H_arr[0] = 1.0 * B_arr[0]

# the gradient vectors over iteration get appended
grad_it_arr = _grad_fn(c_arr_renorm)
grad_it_arr = np.reshape(grad_it_arr, (1, len_c))

# c_arr over iterations
c_arr_it = np.reshape(c_arr_renorm, (1, len_c))

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
    
    print('Calculating gradient...')
    grads = _grad_fn(c_arr_renorm)
    grad_it_arr = np.append(grad_it_arr,
                            np.reshape(grads, (1,len_c)), axis=0)

    print('Calculating hessian...')
    hess = _data_hess_fn(c_arr_renorm)
    # adding some regularization to make it well conditioned
    hess += np.identity(len_c) * 0.1
    H_arr = np.append(H_arr,
                      np.reshape(hess,(1, len_c, len_c)),
                      axis=0)
    hess_inv = jnp.linalg.inv(hess)
    
    print('Updating c_arr_renorm...')
    c_arr_renorm = _update_H(c_arr_renorm, grads, hess_inv)
    c_arr_it = np.append(c_arr_it,
                         np.reshape(c_arr_renorm, (1, len_c)), axis=0)
    
    print('Calculating loss...')
    loss = _loss_fn(c_arr_renorm)

    model_misfit = 0.0
    data_misfit = loss

    loss_diff = loss_prev - loss
    loss_arr.append(loss)
    itercount += 1
    t2 = time.time()
    print(f'[{itercount:3d} | {(t2-t1):6.1f} sec ] data_misfit = {data_misfit:12.5e} loss-diff\
 = {loss_diff:12.5e}; ' +
          f'max-grads = {abs(grads).max():12.5e} model_misfit={model_misfit:12.5e}')

    print('Calculating approximate Hessian...')
    # using BFGS to get approximate hessian
    y_k = grad_it_arr[-1] - grad_it_arr[-2]
    dx_k = c_arr_it[-1] - c_arr_it[-2]
    B_k = B_arr[-1]
    B_arr = np.append(B_arr,
                      np.reshape(_approx_B(B_k, y_k, dx_k), (1, len_c, len_c)),
                      axis=0)

    # sys.exit()

t2s = time.time()




