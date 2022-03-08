import os
import time
from tqdm import tqdm
import argparse
from datetime import date
from datetime import datetime
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mu", help="regularization",
                    type=float, default=1.0)
parser.add_argument("--store_hess", help="store hessians",
                    type=bool, default=False)
parser.add_argument("--read_hess", help="store hessians",
                    type=bool, default=False)
parser.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
parser.add_argument("--batch_run", help="flag to indicate its a batch run",
                    type=int, default=0)
parser.add_argument("--batch_rundir", help="local directory for batch run",
                    type=str, default=".")
parser.add_argument("--s", help="which s is being fit, default is 0 which is all",
                    type=int, default=0)
PARGS = parser.parse_args()
#------------------------------------------------------------------------------

import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
from qdpy import globalvars as gvar_jax
from qdpy import jax_functions as jf
from plotter import postplotter
from plotter import plot_acoeffs_datavsmodel as plot_acoeffs

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

if(not PARGS.batch_run):
    n0lminlmax_dir = f"{package_dir}/dpy_jax"
    outdir = f"{scratch_dir}/dpy_jax"
    summdir = f"{scratch_dir}/summaryfiles"
    plotdir = f"{scratch_dir}/plots"

else:
    n0lminlmax_dir = f"{PARGS.batch_rundir}"
    outdir = f"{PARGS.batch_rundir}"
    summdir = f"{PARGS.batch_rundir}/summaryfiles"
    plotdir = f"{PARGS.batch_rundir}/plots"
    
#------------------------------------------------------------------------# 
ARGS = np.loadtxt(f"{n0lminlmax_dir}/.n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]),
                            relpath=n0lminlmax_dir,
                            instrument=PARGS.instrument,
                            tslen=int(ARGS[6]),
                            daynum=int(ARGS[7]),
                            numsplits=int(ARGS[8]))

soln_summary = {}
soln_summary['params'] = {}
soln_summary['params']['dpy'] = {}
soln_summary['params']['dpy']['n0'] = int(ARGS[0])
soln_summary['params']['dpy']['lmin'] = int(ARGS[1])
soln_summary['params']['dpy']['lmax'] = int(ARGS[2])
soln_summary['params']['dpy']['rth'] = ARGS[3]
soln_summary['params']['dpy']['knot_num'] = int(ARGS[4])
soln_summary['params']['dpy']['GVARS'] = jf.dict2obj(GVARS.__dict__)
#-------------loading precomputed files for the problem-------------------# 
sfx = GVARS.filename_suffix
data = np.load(f'{outdir}/data_model.{sfx}.npy')
true_params_flat = np.load(f'{outdir}/true_params_flat.{sfx}.npy')
param_coeff_flat = np.load(f'{outdir}/param_coeff_flat.{sfx}.npy')
fixed_part = np.load(f'{outdir}/fixed_part.{sfx}.npy')
acoeffs_sigma_HMI = np.load(f'{outdir}/acoeffs_sigma_HMI.{sfx}.npy')
acoeffs_HMI = np.load(f'{outdir}/acoeffs_HMI.{sfx}.npy')
cind_arr = np.load(f'{outdir}/cind_arr.{sfx}.npy')
sind_arr = np.load(f'{outdir}/sind_arr.{sfx}.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load(f'{outdir}/RL_poly.{sfx}.npy')
# model_params_sigma = np.load(f'{outdir}/model_params_sigma.npy')*100.
sigma2scale = np.load(f'{outdir}/sigma2scale.{sfx}.npy')
D_bsp_j_D_bsp_k = np.load(f'{outdir}/D_bsp_j_D_bsp_k.{sfx}.npy')
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

try:
    knee_mu = np.hstack((np.load(f"{PARGS.batch_rundir}/muval.s1.npy"),
                         np.load(f"{PARGS.batch_rundir}/muval.s3.npy"),
                         np.load(f"{PARGS.batch_rundir}/muval.s5.npy")))
    print('Using optimal mu.')
except FileNotFoundError:
    knee_mu = np.array([1.e-3, 1.e-3, 1.e-3])
    print('Not using optimal mu.')


print(f"knee_mu = {knee_mu}")
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

#---------------------------------------------------------------------------
def umax(arr):
    maxval = max(arr)
    minval = min(arr)
    if abs(maxval) >= abs(minval):
        return maxval
    else:
        return minval

def print_info(itercount, tdiff, data_misfit, loss_diff, max_grads, model_misfit):
    print(f'[{itercount:3d} | ' +
          f'{tdiff:6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} ' +
          f'loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {max_grads:12.5e} ' +
          f'model_misfit={model_misfit:12.5e}')
    return None


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


def loop_in_mults(mult_ind, ppa):
    _pred, _pred_acoeff = ppa
    _pred_omega = jdc(_pred, (mult_ind*dim_hyper,), (dim_hyper,))
    _pred_acoeff = jdc_update(_pred_acoeff,
                              ((Pjl[mult_ind] @ _pred_omega)/
                               aconv_denom[mult_ind]),
                              (mult_ind * num_j,))
    return (_pred, _pred_acoeff)

#---------------checking that the loaded data are correct----------------#
pred = fixed_part + true_params_flat @ param_coeff_flat
pred_acoeffs = jnp.zeros(num_j * nmults)
__, pred_acoeffs = foril(0, nmults, loop_in_mults, (pred, pred_acoeffs))

data_acoeffs = jnp.zeros(num_j * nmults)
__, data_acoeffs = foril(0, nmults, loop_in_mults, (data, data_acoeffs))

np.testing.assert_array_almost_equal(pred, data)
print(f"[TESTING] pred = data: PASSED")
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)
print(f"[TESTING] pred_acoeffs = data_acoeffs: PASSED")

#-----------------------------------------------------------------------#
# changing to the HMI acoeffs if doing this for real data 
data_acoeffs = GVARS.acoeffs_true
# np.random.seed(3)
# data_acoeffs_err = np.random.normal(loc=0, scale=acoeffs_sigma_HMI)
# data_acoeffs = data_acoeffs + 1.0*data_acoeffs_err
data_acoeffs_out_HMI = GVARS.acoeffs_out_HMI
print(f"data_acoeffs = {data_acoeffs[:15]}")
#-----------------------------------------------------------------------#

def data_misfit_arr_fn(c_arr, fp, data_acoeffs_iter):
    pred = fp + c_arr @ param_coeff_flat
    pred_acoeffs = jnp.zeros(num_j * nmults)
    __, pred_acoeffs = foril(0, nmults, loop_in_mults, (pred, pred_acoeffs))
    data_misfit_arr = (data_acoeffs_iter - pred_acoeffs)/acoeffs_sigma_HMI
    return data_misfit_arr


def data_misfit_fn(c_arr, fp, data_acoeffs_iter):
    data_misfit_arr = data_misfit_arr_fn(c_arr, fp, data_acoeffs_iter)
    return jnp.sum(jnp.square(data_misfit_arr))


def model_misfit_fn(c_arr, carr_fixed, mu_scale=knee_mu):
    # Djk is the same for s=1, 3, 5
    Djk = D_bsp_j_D_bsp_k
    sidx, eidx = 0, GVARS.knot_ind_th
    cDc = 0.0

    for i in range(len_s):
        carr_padding = carr_fixed[sind_arr[i], sidx:eidx]
        cd = jnp.append(carr_padding, c_arr[i::len_s])
        lambda_factor = jnp.trace(data_hess_dpy[i::len_s, i::len_s])
        lambda_factor /= len_data * len_s
        cDc += mu_scale[i] * cd @ Djk @ cd * lambda_factor
    return cDc


def hessian(f):
    return jacfwd(jacrev(f))


def loss_fn(c_arr, carr_fixed, fp, data_acoeffs_iter):
    data_misfit_val = data_misfit_fn(c_arr, fp, data_acoeffs_iter)
    model_misfit_val = model_misfit_fn(c_arr, carr_fixed)
    total_misfit = data_misfit_val + mu*model_misfit_val
    return total_misfit


def update_cgrad(c_arr, grads, steplen):
    gstr = jnp.sqrt(jnp.sum(jnp.square(grads)))
    return jax.tree_multimap(lambda c, g, sl: c - sl * g / gstr, c_arr, grads, steplen)


def update_H(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)

#----------------------------------------------------------------------# 
# plotting acoeffs pred and data to see if we should expect got fit
plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'ref',
                                      plotdir=plotdir)
plot_acoeffs.plot_acoeffs_dm_scaled(pred_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'ref',
                                    plotdir=plotdir)
# sys.exit()
#----------------------------------------------------------------------# 
# the length of data
len_data = len(data_acoeffs)

# the regularizing parameter
mu = PARGS.mu

#---------------------- jitting the functions --------------------------#
data_hess_fn = hessian(data_misfit_fn)
model_hess_fn = hessian(model_misfit_fn)
grad_fn = jax.grad(loss_fn)
hess_fn = hessian(loss_fn)

_grad_fn = jit(grad_fn)
_hess_fn = jit(hess_fn)
_update_H = jit(update_H)
_update_cgrad = jit(update_cgrad)
_loss_fn = jit(loss_fn)

#------------------------------------------------------------------------
def iterative_RLS(c_arr, carr_fixed, fp, data_iter, iternum=0, lossthr=1e-3):
    N = len(data_acoeffs)
    loss = 1e25
    loss_diff = loss - 1.
    loss_arr = []
    loss_threshold = 1e-12
    maxiter = 20
    itercount = 0
    while ((abs(loss_diff) > loss_threshold) and
        (itercount < maxiter)):
        t1 = time.time()
        loss_prev = loss
        grads = _grad_fn(c_arr, carr_fixed, fp, data_iter)
        c_arr = _update_H(c_arr, grads, hess_inv)
        loss = _loss_fn(c_arr, carr_fixed, fp, data_iter)
        model_misfit = model_misfit_fn(c_arr, carr_fixed)
        data_misfit = loss - mu*model_misfit

        loss_diff = loss_prev - loss
        loss_arr.append(loss)
        itercount += 1
        t2 = time.time()
        # print_info(itercount, t2-t1,
        #            data_misfit, loss_diff, abs(grads).max(), model_misfit)
        t2 = time.time()
    t2s = time.time()
    # print(f"-------------------------------------------------------------------------")
    return c_arr
    

#-----------------------the main training loop--------------------------#
# initialization of params
c_init = np.ones_like(true_params_flat) * true_params_flat
print(f"Number of parameters = {len(c_init)}")

#------------------plotting the initial profiles-------------------#                     
c_arr_init_full = jf.c4fit_2_c4plot(GVARS, c_init,
                                    sind_arr, cind_arr)

# converting ctrl points to wsr and plotting
ctrl_zero_error = np.zeros_like(c_arr_init_full)
init_plot = postplotter.postplotter(GVARS, c_arr_init_full,
                                    ctrl_zero_error, 'init',
                                    plotdir=plotdir)
#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
init_acoeffs = data_misfit_arr_fn(c_init, fixed_part, data_acoeffs)*acoeffs_sigma_HMI +\
               data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'init',
                                      plotdir=plotdir)
plot_acoeffs.plot_acoeffs_dm_scaled(init_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'init',
                                    plotdir=plotdir)
#----------------------------------------------------------------------#
N = len(data_acoeffs)
loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-12
maxiter = 20
N0 = 3
itercount = 0

hsuffix = f"{int(ARGS[4])}s.{GVARS.eigtype}"
print(hsuffix, f"{outdir}/dhess.{hsuffix}.{sfx}.npy")
if PARGS.read_hess:
    data_hess_dpy = np.load(f"{outdir}/dhess.{hsuffix}.{sfx}.npy")
    model_hess_dpy = np.load(f"{outdir}/mhess.{hsuffix}.{sfx}.npy")
else:
    data_hess_dpy = jnp.asarray(data_hess_fn(c_init, fixed_part, data_acoeffs))
    model_hess_dpy = jnp.asarray(model_hess_fn(c_init, GVARS.ctrl_arr_dpt_full))

total_hess = data_hess_dpy + mu*model_hess_dpy
hess_inv = jnp.linalg.inv(total_hess)


if PARGS.store_hess:
    np.save(f"{outdir}/dhess.{hsuffix}.{sfx}.npy", data_hess_dpy)
    np.save(f"{outdir}/mhess.{hsuffix}.{sfx}.npy", model_hess_dpy)

tdiff = 0
grads = _grad_fn(c_init, GVARS.ctrl_arr_dpt_full, fixed_part, data_acoeffs)
loss = _loss_fn(c_init, GVARS.ctrl_arr_dpt_full, fixed_part, data_acoeffs)
model_misfit = model_misfit_fn(c_init, GVARS.ctrl_arr_dpt_full)
data_misfit = loss - mu*model_misfit
print_info(itercount, tdiff, data_misfit, loss_diff, abs(grads).max(), model_misfit)
steplen = 1.0e-2
t1s = time.time()

# print(f"grads = {grads[:10]}")
# print(f"grad @ hess_inv = {(grads @ hess_inv)[:10]}")

data_acoeffs_iter = data_acoeffs*1.0
c_arr_allk = [c_init]
kiter = 0
kmax = 8
delta_k = 100000

print(f"-----------------BEFORE FITTING ---------------------")
for i in range(len_s):
    s = 2*sind_arr[i] + 1
    print(f"   a{s} = {data_acoeffs_iter[sind_arr[i]::3][:6]}")
print(f"-----------------------------------------------------")
c_arr = iterative_RLS(c_init, GVARS.ctrl_arr_dpt_full, fixed_part, data_acoeffs_iter)
data_acoeffs_iter = data_misfit_arr_fn(c_arr, fixed_part,
                                       data_acoeffs_iter)*acoeffs_sigma_HMI
prntarr = c_arr[0::len_s]/true_params_flat[0::len_s]
for i in range(len_s):
    s = 2*sind_arr[i] + 1
    print(f"   c{s} = {prntarr[i::len_s][:6]}")
    print(f"   a{s} = {data_acoeffs_iter[sind_arr[i]::3][:6]}")

ctot_local = c_arr * 1.0
c_arr_total = c_arr * 1.0

while(kiter < kmax):
    for ii in tqdm(range(2**kiter * N0), desc=f"k={kiter}"):
        c_arr = 0.0 * c_init
        c_arr = iterative_RLS(c_arr, GVARS.ctrl_arr_dpt_full*0.0,
                              fixed_part*0.0, data_acoeffs_iter, iternum=2**kiter+1)
        c_arr_total += c_arr
        ctot_local += c_arr
        data_acoeffs_iter = data_misfit_arr_fn(c_arr, fixed_part*0.0,
                                               data_acoeffs_iter)*acoeffs_sigma_HMI

    prntarr = c_arr_total/true_params_flat
    for i in range(len_s):
        s = 2*sind_arr[i] + 1
        print(f"   c{s} = {prntarr[i::len_s][:6]}")
        print(f"   a{s} = {data_acoeffs_iter[i::len_s][:6]}")
    c_arr_allk.append(ctot_local)
    kiter += 1
    ctot_local = 0.0

    diff_ratio = (c_arr_allk[-1] - c_arr_allk[-2])/true_params_flat
    print(f"[{kiter}] --- delta_k_old = {max(abs(diff_ratio*true_params_flat))}")
    diff_ratio_s = [diff_ratio[i::len_s] for i in range(len_s)]
    delta_k = [umax(diff_ratio_s[i]) for i in range(len_s)]
    print(f"[{kiter}] --- delta_k_new = {delta_k}")
    print(f"-----------------------------------------------------")


#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
final_acoeffs = data_misfit_arr_fn(c_arr, fixed_part,
                                   data_acoeffs)*acoeffs_sigma_HMI + data_acoeffs

plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs, data_acoeffs,
                                      data_acoeffs_out_HMI,
                                      acoeffs_sigma_HMI, 'final',
                                      plotdir=plotdir)

plot_acoeffs.plot_acoeffs_dm_scaled(final_acoeffs, data_acoeffs,
                                    data_acoeffs_out_HMI,
                                    acoeffs_sigma_HMI, 'final',
                                    plotdir=plotdir)
#----------------------------------------------------------------------# 
np.save(f"{outdir}/carr_iterative_{mu:.1e}_{kmax}k.npy", c_arr_total)
c_arr_fit = c_arr_total/true_params_flat

# finding chisq
total_misfit = data_misfit_arr_fn(c_arr_total, fixed_part,
                                  data_acoeffs)
# degrees of freedom
dof = len(pred_acoeffs) - len(c_arr_total)
chisq = total_misfit/dof

for i in range(len_s):
    print(c_arr_fit[i::len_s])

#------------------plotting the post fitting profiles-------------------#
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                   sind_arr, cind_arr)

# converting ctrl points to wsr and plotting
fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full,
                                   ctrl_zero_error, 'fit-iter',
                                   plotdir=plotdir)
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

soln_summary['c_arr_fit'] = c_arr_fit
soln_summary['true_params_flat'] = true_params_flat
soln_summary['cind_arr'] = cind_arr
soln_summary['sind_arr'] = sind_arr

soln_summary['acoeff'] = {}
soln_summary['acoeff']['fit'] = final_acoeffs
soln_summary['acoeff']['data'] = data_acoeffs
soln_summary['acoeff']['sigma'] = acoeffs_sigma_HMI

soln_summary['data_hess'] = data_hess_dpy
soln_summary['model_hess'] = model_hess_dpy
soln_summary['loss_arr'] = loss_arr
soln_summary['mu'] = mu
soln_summary['knee_mu'] = knee_mu
soln_summary['chisq'] = chisq

todays_date = date.today()
timeprefix = datetime.now().strftime("%H.%M")
dateprefix = f"{todays_date.day:02d}.{todays_date.month:02d}.{todays_date.year:04d}"
fsuffix = f"{dateprefix}-{timeprefix}-{hsuffix}-{PARGS.s}"
jf.save_obj(soln_summary, f"{summdir}/summary.dpt.iterative-{fsuffix}")

