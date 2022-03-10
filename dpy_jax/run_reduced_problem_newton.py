import os
import time
import argparse
from datetime import date
from datetime import datetime
#-----------------------------------------------------------------------#
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
parser.add_argument("--mu_batchdir", help="directory of converged mu",
                    type=str, default=".")
parser.add_argument("--plot", help="plot",
                    type=bool, default=False)
parser.add_argument("--s", help="which s is being fit, default is 0 which is all",
                    type=int, default=0)
PARGS = parser.parse_args()
#------------------------------------------------------------------------# 
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import arviz as az
import sys
#------------------------------------------------------------------------# 
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
#----------------- import local modules -----------------------------#
from qdpy import globalvars as gvar_jax
from qdpy import jax_functions as jf
from plotter import postplotter
from plotter import plot_acoeffs_datavsmodel as plot_acoeffs
#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

if (not PARGS.batch_run):
    n0lminlmax_dir = f"{package_dir}/dpy_jax"
    outdir = f"{scratch_dir}/dpy_jax"
    summdir = f"{scratch_dir}/summaryfiles"
    plotdir = f"{scratch_dir}/plots"
    knee_mu = np.array([1., 1., 1.])

else:
    n0lminlmax_dir = f"{PARGS.batch_rundir}"
    outdir = f"{PARGS.batch_rundir}"
    summdir = f"{PARGS.batch_rundir}/summaryfiles"
    plotdir = f"{PARGS.batch_rundir}/plots"
    try:
        knee_mu = np.hstack((np.load(f"{PARGS.mu_batchdir}/muval.s1.npy"),
                             np.load(f"{PARGS.mu_batchdir}/muval.s3.npy"),
                             np.load(f"{PARGS.mu_batchdir}/muval.s5.npy")))
        knee_mu *= 100.
    except FileNotFoundError:
        knee_mu = np.array([1., 1., 1.])
        found_optimal = False

print(f"outdir = {outdir}")
print(f"knee_mu = {knee_mu}")
#----------------------------------------------------------------------#
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
len_s = len(sind_arr) # number of s to fit
nc = len(cind_arr) # number of c to fit

#------------------------------------------------------------------------#
# slicing the Pjl correctly in angular degree s and computing normalization
# calculating the denominator of a-coefficient converion apriori
# shape (nmults, num_j)
Pjl = RL_poly[:, smin:smax+1:2, :]
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
#-------------------------------------------------------------------------#
def print_info(itercount, tdiff, data_misfit, loss_diff, max_grads, model_misfit):
    print(f'[{itercount:3d} | ' +
          f'{tdiff:6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} ' +
          f'loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {max_grads:12.5e} ' +
          f'model_misfit={model_misfit:12.5e}')
    return None


def loop_in_mults(mult_ind, ppa):
    _pred, _pred_acoeff = ppa
    _pred_omega = jdc(_pred, (mult_ind*dim_hyper,), (dim_hyper,))
    _pred_acoeff = jdc_update(_pred_acoeff,
                              ((Pjl[mult_ind] @ _pred_omega)/
                               aconv_denom[mult_ind]),
                              (mult_ind * num_j,))
    return (_pred, _pred_acoeff)


def data_misfit_arr_fn(c_arr):
    pred = fixed_part + c_arr @ param_coeff_flat
    pred_acoeffs = jnp.zeros(num_j * nmults)
    __, pred_acoeffs = foril(0, nmults, loop_in_mults, (pred, pred_acoeffs))
    data_misfit_arr = (data_acoeffs - pred_acoeffs)/acoeffs_sigma_HMI
    return data_misfit_arr


def data_misfit_fn(c_arr):
    data_misfit_arr = data_misfit_arr_fn(c_arr)
    return jnp.sum(jnp.square(data_misfit_arr))


def get_pc_pjl(c_arr):
    pred = param_coeff_flat
    pc_pjl = jnp.zeros((len(c_arr), num_j * nmults))

    def loop_in_c(cind, pc):
        def loop_in_mults2(mult_ind, pred_acoeff):
            pred_omega = jdc(pred[cind], (mult_ind*dim_hyper,), (dim_hyper,))
            pred_acoeff = jdc_update(pred_acoeff,
                                    (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                    (mult_ind * num_j,))
            return pred_acoeff

        pcpjl = foril(0, nmults, loop_in_mults2, pc[cind])
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

def get_GT_Cd_inv(c_arr):
    '''Function to compute G.T @ Cd_inv
    for obtaining the G^{-g} later.'
    '''
    pc_pjl = get_pc_pjl(c_arr)
    GT_Cd_inv= pc_pjl @ np.diag(1/acoeffs_sigma_HMI**2)
    return GT_Cd_inv


def model_misfit_fn(c_arr, mu_scale=knee_mu):
    # Djk is the same for s=1, 3, 5
    Djk = D_bsp_j_D_bsp_k
    sidx, eidx = 0, GVARS.knot_ind_th
    cDc = 0.0

    for i in range(len_s):
        carr_padding = GVARS.ctrl_arr_dpt_full[sind_arr[i], sidx:eidx]
        cd = jnp.append(carr_padding, c_arr[i::len_s])
        lambda_factor = jnp.trace(data_hess_dpy[i::len_s, i::len_s])
        lambda_factor /= len_data * len_s
        cDc += mu_scale[i] * cd @ Djk @ cd * lambda_factor
    return cDc


def hessian(f):
    return jacfwd(jacrev(f))


def loss_fn(c_arr):
    data_misfit_val = data_misfit_fn(c_arr)
    model_misfit_val = model_misfit_fn(c_arr)
    total_misfit = data_misfit_val + mu*model_misfit_val
    return total_misfit


def update_cgrad(c_arr, grads, steplen):
    gstr = jnp.sqrt(jnp.sum(jnp.square(grads)))
    return jax.tree_multimap(lambda c, g, sl: c - sl * g / gstr,
                             c_arr, grads, steplen)


def update_H(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h,
                             c_arr, grads, hess_inv)


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

#---------------------- setting data-acoeffs ---------------------------#
# changing to the HMI acoeffs if doing this for real data 
data_acoeffs = GVARS.acoeffs_true
# np.random.seed(3)
# data_acoeffs_err = np.random.normal(loc=0, scale=acoeffs_sigma_HMI)
# data_acoeffs = data_acoeffs + 0.0*data_acoeffs_err
data_acoeffs_out_HMI = GVARS.acoeffs_out_HMI
print(f"data_acoeffs = {data_acoeffs[:15]}")
#----------------------------------------------------------------------# 
if PARGS.plot:
    # plotting acoeffs pred and data to see if we should expect got fit
    plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs, data_acoeffs,
                          data_acoeffs_out_HMI,
                          acoeffs_sigma_HMI, 'ref',
                          plotdir=plotdir)
    plot_acoeffs.plot_acoeffs_dm_scaled(pred_acoeffs, data_acoeffs,
                        data_acoeffs_out_HMI,
                        acoeffs_sigma_HMI, 'ref',
                        plotdir=plotdir)
#----------------------------------------------------------------------# 
len_data = len(data_acoeffs) # length of data
mu = PARGS.mu # regularization parameter

# changing the regularization as a function of depth
# mu_depth = np.zeros_like(GVARS.ctrl_arr_dpt_full[0, :])
# rth_soft = GVARS.rth + 0.01
# width = 0.003
# mu_depth = 0.5 * (1 - np.tanh((GVARS.knot_locs - rth_soft)/width))
# mu_depth = 1e10 * jnp.sqrt(jnp.asarray(mu_depth)) / mu
#-----------------------the main training loop--------------------------#
# initialization of params
c_init = np.ones_like(true_params_flat) + 0.0*np.random.rand(len(true_params_flat))
c_init *= true_params_flat
print(f"Number of parameters = {len(c_init)}")

#------------------plotting the initial profiles-------------------#                     
if PARGS.plot:
    c_arr_init_full = jf.c4fit_2_c4plot(GVARS, c_init,
                                        sind_arr, cind_arr)

    # converting ctrl points to wsr and plotting
    ctrl_zero_error = np.zeros_like(c_arr_init_full)
    init_plot = postplotter.postplotter(GVARS, c_arr_init_full, ctrl_zero_error, 'init',
                                        plotdir=plotdir)
#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
pred_init = fixed_part + c_init @ param_coeff_flat
init_acoeffs = jnp.zeros(num_j * nmults)
__, init_acoeffs = foril(0, nmults, loop_in_mults, (pred_init, init_acoeffs))

if PARGS.plot:
    plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs, data_acoeffs,
                                          data_acoeffs_out_HMI,
                                          acoeffs_sigma_HMI, 'init',
                                          plotdir=plotdir)
    plot_acoeffs.plot_acoeffs_dm_scaled(init_acoeffs, data_acoeffs,
                                        data_acoeffs_out_HMI,
                                        acoeffs_sigma_HMI, 'init',
                                        plotdir=plotdir)
#----------------------------------------------------------------------#
# print(f"mu depth shape = {mu_depth.shape}")
print(f"ctrl full shape = {GVARS.ctrl_arr_dpt_full.shape}")
N = len(data_acoeffs)
loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-12
maxiter = 20
itercount = 0

hsuffix = f"{int(ARGS[4])}s.{GVARS.eigtype}"
print(hsuffix)
if PARGS.read_hess:
    data_hess_dpy = np.load(f"{outdir}/dhess.{hsuffix}.{sfx}.npy")
    model_hess_dpy = np.load(f"{outdir}/mhess.{hsuffix}.{sfx}.npy")
else:
    data_hess_dpy = jnp.asarray(data_hess_fn(c_init))
    model_hess_dpy = jnp.asarray(model_hess_fn(c_init))

total_hess = data_hess_dpy + mu*model_hess_dpy
hess_inv = jnp.linalg.inv(total_hess)


if PARGS.store_hess:
    np.save(f"{outdir}/dhess.{hsuffix}.{sfx}.npy", data_hess_dpy)
    np.save(f"{outdir}/mhess.{hsuffix}.{sfx}.npy", model_hess_dpy)


tdiff = 0
grads = _grad_fn(c_init)
loss = _loss_fn(c_init)
model_misfit = model_misfit_fn(c_init)
data_misfit = loss - mu*model_misfit
print_info(itercount, tdiff, data_misfit, loss_diff, abs(grads).max(), model_misfit)
c_arr = c_init
steplen = 1.0e-2

t1s = time.time()
while ((abs(loss_diff) > loss_threshold) and
       (itercount < maxiter)):
    t1 = time.time()
    loss_prev = loss
    grads = _grad_fn(c_arr)
    c_arr = _update_H(c_arr, grads, hess_inv)
    # c_arr = _update_cgrad(c_arr, grads, steplen)
    loss = _loss_fn(c_arr)
    if loss > loss_prev:
        steplen /= 0.5

    model_misfit = model_misfit_fn(c_arr)
    data_misfit = loss - mu*model_misfit

    loss_diff = loss_prev - loss
    loss_arr.append(loss)
    itercount += 1
    t2 = time.time()
    print_info(itercount, t2-t1, data_misfit, loss_diff, abs(grads).max(), model_misfit)

    t2 = time.time()

t2s = time.time()
print(f"Total time taken = {(t2s-t1s):12.3f} seconds")

total_misfit = data_misfit_fn(c_arr)
num_data = len(pred_acoeffs)
chisq = total_misfit/num_data
print(f"chisq = {chisq:.5f}")

#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
final_acoeffs = data_misfit_arr_fn(c_arr)*acoeffs_sigma_HMI + data_acoeffs

if PARGS.plot:
    plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs, data_acoeffs,
                                          data_acoeffs_out_HMI,
                                          acoeffs_sigma_HMI, 'final',
                                          plotdir=plotdir)
    plot_acoeffs.plot_acoeffs_dm_scaled(final_acoeffs, data_acoeffs,
                                        data_acoeffs_out_HMI,
                                        acoeffs_sigma_HMI, 'final',
                                        plotdir=plotdir)
#----------------------------------------------------------------------# 
# reconverting back to model_params in units of true_params_flat
c_arr_fit = c_arr/true_params_flat
np.save(f"{outdir}/carr_fit_{mu:.5e}.npy", c_arr)
print(f"carrfit saved in: {outdir}/carr_fit_{mu:.5e}.npy")

for i in range(len_s):
    print(c_arr_fit[i::len_s])

#------------------------------------------------------------------------#
with open(f"{current_dir}/reg_misfit.txt", "a") as f:
    f.seek(0, os.SEEK_END)
    opstr = f"{mu:18.12e}, {data_misfit:18.12e}, {model_misfit:18.12e}\n"
    f.write(opstr)

#-----------------finding the model covariance matrix------------------#
# can be shown that the model covariance matrix has the following form
# C_m = G^{-g} @ C_d @ G^{-g}.T
# G^{-g} = total_hess_inv @ G.T @ C_d_inv

GT_Cd_inv = get_GT_Cd_inv(c_arr)
G_g_inv = hess_inv @ GT_Cd_inv
C_d = jnp.diag(acoeffs_sigma_HMI**2)
C_m = jf.get_model_covariance(G_g_inv, C_d)
ctrl_arr_err = jnp.sqrt(jnp.diag(C_m))

#------------------plotting the post fitting profiles-------------------#
if PARGS.plot:
    c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                       sind_arr, cind_arr)

    # making the full error array to pass into c4fit_2_c4plot
    ctrl_arr_err_full = np.zeros_like(c_arr_fit_full)
    ctrl_arr_err_full[sind_arr, -len(true_params_flat)//len_s:] =\
                                jnp.reshape(ctrl_arr_err, (len_s, -1), 'F')
    c_arr_err_full = jnp.reshape(ctrl_arr_err_full, (len_s, -1), 'F')

    # converting ctrl points to wsr and plotting
    fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full, ctrl_arr_err_full, 'fit',
                                       plotdir=plotdir)

'''
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
'''

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
if(not PARGS.store_hess and not PARGS.batch_run):
    jf.save_obj(soln_summary, f"{summdir}/summary.dpt-{fsuffix}")
