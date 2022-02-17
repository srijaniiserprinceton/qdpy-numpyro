import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mu", help="regularization",
                    type=float, default=0.)
parser.add_argument("--store_hess", help="store hessians",
                    type=bool, default=False)
parser.add_argument("--read_hess", help="store hessians",
                    type=bool, default=False)
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
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
outdir = f"{scratch_dir}/dpy_jax"
plotdir = f"{scratch_dir}/plots"
sys.path.append(f"{package_dir}/plotter")
import postplotter
import plot_acoeffs_datavsmodel as plot_acoeffs
#------------------------------------------------------------------------# 
ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]),
                            relpath=outdir)
#-------------loading precomputed files for the problem-------------------# 
data = np.load(f'{outdir}/data_model.npy')
true_params_flat = np.load(f'{outdir}/true_params_flat.npy')
param_coeff_flat = np.load(f'{outdir}/param_coeff_flat.npy')
fixed_part = np.load(f'{outdir}/fixed_part.npy')
acoeffs_sigma_HMI = np.load(f'{outdir}/acoeffs_sigma_HMI.npy')
acoeffs_HMI = np.load(f'{outdir}/acoeffs_HMI.npy')
cind_arr = np.load(f'{outdir}/cind_arr.npy')
sind_arr = np.load(f'{outdir}/sind_arr.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load(f'{outdir}/RL_poly.npy')
# model_params_sigma = np.load(f'{outdir}/model_params_sigma.npy')*100.
sigma2scale = np.load(f'{outdir}/sigma2scale.npy')
D_bsp_j_D_bsp_k = np.load(f'{outdir}/D_bsp_j_D_bsp_k.npy')
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

mu_scaling = np.array([1., 1., 1.])
# knee_mu = np.array([2.15443e-5, 1.59381e-7, 1.29155e-7])
# mu_scaling = knee_mu/knee_mu[0]

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
# adding the contribution from the fitting part
# and testing that arrays are very close
pred = fixed_part * 1.0
pred += true_params_flat @ param_coeff_flat
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
np.random.seed(3)
data_acoeffs_err = np.random.normal(loc=0, scale=acoeffs_sigma_HMI)
data_acoeffs = data_acoeffs + 1.0*data_acoeffs_err
data_acoeffs_out_HMI = GVARS.acoeffs_out_HMI
print(f"data_acoeffs = {data_acoeffs[:15]}")

#----------------------------------------------------------------------# 
# plotting acoeffs pred and data to see if we should expect got fit
plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'ref')
plot_acoeffs.plot_acoeffs_dm_scaled(pred_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'ref')
# sys.exit()
#----------------------------------------------------------------------# 
# the length of data
len_data = len(data_acoeffs)

# the regularizing parameter
mu = PARGS.mu


# changing the regularization as a function of depth
mu_depth = np.zeros_like(GVARS.ctrl_arr_dpt_full[0,:])
rth_soft = 0.80
width = 0.003
mu_depth = 0.5 * (1 - np.tanh((GVARS.knot_locs - rth_soft)/width))
mu_depth = 1e10 * jnp.sqrt(jnp.asarray(mu_depth)) / mu


def print_info(itercount, tdiff, data_misfit, loss_diff, max_grads, model_misfit):
    print(f'[{itercount:3d} | ' +
          f'{tdiff:6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} ' +
          f'loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {max_grads:12.5e} ' +
          f'model_misfit={model_misfit:12.5e}')
    return None


# the model function that is used by MCMC kernel
def data_misfit_fn(c_arr, data_acoeffs_iter=data_acoeffs):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)

    # denormalizing to make actual model params
    # c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    c_arr_denorm = c_arr
    pred = fixed_part + c_arr_denorm @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    data_misfit_arr = (pred_acoeffs - data_acoeffs_iter)/acoeffs_sigma_HMI

    return jnp.sum(jnp.square(data_misfit_arr))


def get_pc_pjl(c_arr):
    pred = param_coeff_flat
    pc_pjl = jnp.zeros((len(c_arr), num_j * nmults))

    def loop_in_c(cind, pc):
        def loop_in_mults(mult_ind, pred_acoeff):
            pred_omega = jdc(pred[cind], (mult_ind*dim_hyper,), (dim_hyper,))
            pred_acoeff = jdc_update(pred_acoeff,
                                    (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                    (mult_ind * num_j,))
            return pred_acoeff

        pcpjl = foril(0, nmults, loop_in_mults, pc[cind])
        pc = jidx_update(pc,
                         jidx[cind, :],
                         pcpjl)
        return pc

    pc_pjl = foril(0, len(c_arr), loop_in_c, pc_pjl)
    return pc_pjl


def get_dhess_exact(c_arr):
    pc_pjl = get_pc_pjl(c_arr)
    gtg = pc_pjl @ (np.diag(1/acoeffs_sigma_HMI**2) @ pc_pjl.T)
    return 2*gtg


def data_misfit_arr_fn(c_arr, data_acoeffs_iter=data_acoeffs):
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)

    # denormalizing to make actual model params
    # c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    c_arr_denorm = c_arr
    pred = fixed_part + c_arr_denorm @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    data_misfit_arr = (-pred_acoeffs + data_acoeffs_iter)/acoeffs_sigma_HMI

    return data_misfit_arr

def model_misfit_fn(c_arr, mu_scale=mu_scaling):
    # c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    # Djk is the same for s=3 and s=5

    # c_arr_renorm = jf.model_renorm(c_arr, true_params_flat, sigma2scale)
    c_arr_renorm = c_arr
    cd = []
    lambda_factor = []
    start_idx = 0 #max(GVARS.knot_ind_th-4, 0)
    end_idx = GVARS.knot_ind_th
    carr_padding = []
    for i in range(len_s):
        carr_padding.append(GVARS.ctrl_arr_dpt_full[sind_arr[i], start_idx:end_idx])

    for i in range(len_s):
        cd.append(jnp.append(carr_padding[i], c_arr_renorm[i::len_s]))
        lambda_factor.append(jnp.trace(data_hess_dpy[i::len_s, i::len_s]) / len_data * len_s)

    Djk = D_bsp_j_D_bsp_k #[0::len_s, 0::len_s]
    cDc = 0.0

    for i in range(len_s):
        cDc += mu_scale[i] * cd[i] @ Djk @ cd[i] * lambda_factor[i]
        cDc += jnp.sum(mu_depth * jnp.square(cd[i] - GVARS.ctrl_arr_dpt_full[sind_arr[i]]))
    return cDc


def modelcorr_misfit_fn(c_arr, mu_scale=mu_scaling):
    # c_arr_denorm = jf.model_denorm(c_arr, true_params_flat, sigma2scale)
    # Djk is the same for s=3 and s=5

    # c_arr_renorm = jf.model_renorm(c_arr, true_params_flat, sigma2scale)
    c_arr_renorm = c_arr
    cd = []
    lambda_factor = []
    start_idx = 0 #max(GVARS.knot_ind_th-4, 0)
    end_idx = GVARS.knot_ind_th
    carr_padding = []
    for i in range(len_s):
        carr_padding.append(GVARS.ctrl_arr_dpt_full[sind_arr[i], start_idx:end_idx])

    for i in range(len_s):
        cd.append(jnp.append(carr_padding[i], c_arr_renorm[i::len_s]))
        lambda_factor.append(jnp.trace(data_hess_dpy[i::len_s, i::len_s]) / len_data * len_s)
    Djk = D_bsp_j_D_bsp_k #[0::len_s, 0::len_s]
    cDc = 0.0

    for i in range(len_s):
        cDc += mu_scale[i] * cd[i] @ Djk @ cd[i] * lambda_factor[i]
        cDc += jnp.sum(mu_depth * jnp.square(cd[i]))
    return cDc



def hessian(f):
    return jacfwd(jacrev(f))

data_hess_fn = hessian(data_misfit_fn)
model_hess_fn = hessian(model_misfit_fn)

def loss_fn(c_arr, data_acoeffs_iter=data_acoeffs):
    data_misfit_val = data_misfit_fn(c_arr, data_acoeffs_iter)
    model_misfit_val = model_misfit_fn(c_arr)

    # total misfit
    misfit = data_misfit_val + mu*model_misfit_val
    return misfit

def losscorr_fn(c_arr, data_acoeffs_iter=data_acoeffs):
    data_misfit_val = data_misfit_fn(c_arr, data_acoeffs_iter)
    model_misfit_val = modelcorr_misfit_fn(c_arr)

    # total misfit
    misfit = data_misfit_val + mu*model_misfit_val
    return misfit


grad_fn = jax.grad(loss_fn)
gradcorr_fn = jax.grad(losscorr_fn)
hess_fn = hessian(loss_fn)

def update_cgrad(c_arr, grads, steplen):
    gstr = jnp.sqrt(jnp.sum(jnp.square(grads)))
    return jax.tree_multimap(lambda c, g, sl: c - sl * g / gstr, c_arr, grads, steplen)

def update_H(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)


#---------------------- jitting the functions --------------------------#
_grad_fn = jit(grad_fn)
_gradcorr_fn = jit(gradcorr_fn)
_hess_fn = jit(hess_fn)
_update_H = jit(update_H)
_update_cgrad = jit(update_cgrad)
_loss_fn = jit(loss_fn)
_losscorr_fn = jit(losscorr_fn)

#------------------------------------------------------------------------
def iterative_RLS(c_arr, data_iter, iternum=0):
    N = len(data_acoeffs)
    loss = 1e25
    loss_diff = loss - 1.
    loss_arr = []
    loss_threshold = 1e-3
    maxiter = 20
    itercount = 0
    while ((abs(loss_diff) > loss_threshold) and
        (itercount < maxiter)):
        t1 = time.time()
        loss_prev = loss
        if iternum == 0:
            grads = _grad_fn(c_arr, data_iter)
            c_arr = _update_H(c_arr, grads, hess_inv)
            loss = _loss_fn(c_arr, data_iter)
            model_misfit = model_misfit_fn(c_arr)
        else:
            grads = _gradcorr_fn(c_arr, data_iter)
            c_arr = _update_H(c_arr, grads, hess_inv)
            loss = _losscorr_fn(c_arr, data_iter)
            model_misfit = modelcorr_misfit_fn(c_arr)

        data_misfit = loss - mu*model_misfit

        loss_diff = loss_prev - loss
        loss_arr.append(loss)
        itercount += 1
        t2 = time.time()
        print_info(itercount, t2-t1,
                   data_misfit, loss_diff, abs(grads).max(), model_misfit)
        t2 = time.time()
    t2s = time.time()
    print(f"-------------------------------------------------------------------------")
    return c_arr
    #



#-----------------------the main training loop--------------------------#
# initialization of params
# c_init = np.random.uniform(5.0, 20.0, size=len(true_params_flat))*1e-4
# c_init += np.random.rand(len(c_init))
c_init = np.zeros_like(true_params_flat)
print(f"Number of parameters = {len(c_init)}")

#------------------plotting the initial profiles-------------------#                     
c_arr_init_full = jf.c4fit_2_c4plot(GVARS, c_init,
                                    sind_arr, cind_arr)

# converting ctrl points to wsr and plotting
init_plot = postplotter.postplotter(GVARS, c_arr_init_full, 'init')
#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
init_acoeffs = data_misfit_arr_fn(c_init)*acoeffs_sigma_HMI +\
               data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'init')
plot_acoeffs.plot_acoeffs_dm_scaled(init_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'init')
#----------------------------------------------------------------------#
N = len(data_acoeffs)
loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-12
maxiter = 20
N0 = 5
itercount = 0

hsuffix = f"{int(ARGS[4])}s.{GVARS.eigtype}.{GVARS.tslen}d.npy"
print(hsuffix)
if PARGS.read_hess:
    data_hess_dpy = np.load(f"{outdir}/dhess.{hsuffix}")
    model_hess_dpy = np.load(f"{outdir}/mhess.{hsuffix}")
else:
    data_hess_dpy = jnp.asarray(data_hess_fn(c_init))
    model_hess_dpy = jnp.asarray(model_hess_fn(c_init))

total_hess = data_hess_dpy + mu*model_hess_dpy
hess_inv = jnp.linalg.inv(total_hess)


if PARGS.store_hess:
    np.save(f"{outdir}/dhess.{hsuffix}", data_hess_dpy)
    np.save(f"{outdir}/mhess.{hsuffix}", model_hess_dpy)


tdiff = 0
grads = _grad_fn(c_init)
loss = _loss_fn(c_init)
model_misfit = model_misfit_fn(c_init)
data_misfit = loss - mu*model_misfit
print_info(itercount, tdiff, data_misfit, loss_diff, abs(grads).max(), model_misfit)
steplen = 1.0e-2

t1s = time.time()

data_acoeffs_iter = data_acoeffs*1.0
c_arr_allk = [c_init]
kiter = 0
delta_k = 100000

print(f"a5 = {data_acoeffs_iter[0::3][:10]}")
c_arr = iterative_RLS(c_init, data_acoeffs_iter)
data_acoeffs_iter = data_misfit_arr_fn(c_arr, data_acoeffs_iter)*acoeffs_sigma_HMI
prntarr = c_arr[2::3]/true_params_flat[2::3]
print(f"c5 = {prntarr[:10]}")
print(f"a5 = {data_acoeffs_iter[0::3][:10]}")

ctot_local = c_arr
c_arr_total = c_arr
# sys.exit()

while(kiter < 4):
    c_arr = 1.0 * c_init
    for ii in range(2**kiter * N0):
        c_arr = iterative_RLS(c_arr, data_acoeffs_iter, iternum=2**kiter+1)
        c_arr_total += c_arr
        ctot_local += c_arr
        data_acoeffs_iter = data_misfit_arr_fn(c_arr_total, data_acoeffs)*acoeffs_sigma_HMI
        prntarr = c_arr_total[2::3]/true_params_flat[2::3]
        print(f"c5 = {prntarr[:10]}")
        print(f"a5 = {data_acoeffs_iter[0::3][:10]}")
    c_arr_allk.append(ctot_local)
    kiter += 1
    ctot_local = 0.0

    delta_k = max(abs(c_arr_allk[-1] - c_arr_allk[-2]))
    print(f"[{kiter}] --- delta_k = {delta_k}")


#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
final_acoeffs = data_misfit_arr_fn(c_arr)*acoeffs_sigma_HMI + data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'final')

plot_acoeffs.plot_acoeffs_dm_scaled(final_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'final')
#----------------------------------------------------------------------# 

# reconverting back to model_params in units of true_params_flat
# c_arr_fit = jf.model_denorm(c_arr, true_params_flat, sigma2scale)\
#             /true_params_flat
c_arr_fit = sum(c_arr_allk)/true_params_flat

for i in range(len_s):
    print(c_arr_fit[i::len_s])

#------------------plotting the post fitting profiles-------------------#
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                   sind_arr, cind_arr)

# converting ctrl points to wsr and plotting
fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full, 'fit')

#------------------------------------------------------------------------#
with open(f"{current_dir}/reg_misfit.txt", "a") as f:
    f.seek(0, os.SEEK_END)
    opstr = f"{mu:18.12e}, {data_misfit:18.12e}, {model_misfit:18.12e}\n"
    f.write(opstr)

#------------------------------------------------------------------------# 
# plotting the hessians for analysis
fig, ax = plt.subplots(1, 2, figsize=(10,5))

im1 = ax[0].pcolormesh(total_hess)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax[1].pcolormesh(data_hess_dpy)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

plt.tight_layout()
plt.savefig(f'{plotdir}/hessians.png')
plt.close()
