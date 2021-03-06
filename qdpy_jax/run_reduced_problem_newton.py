import os
import time
import argparse

#--------------------- argument parser ----------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--mu", help="regularization",
                    type=float, default=0.)
parser.add_argument("--store_hess", help="store hessians",
                    type=bool, default=False)
parser.add_argument("--read_hess", help="store hessians",
                    type=bool, default=False)
PARGS = parser.parse_args()
#------------------------------------------------------------------------# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse as sp_sparse
import sys
#----------------------JAX related modules-------------------------------#
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

#----------------------local modules------------------------------------#
from qdpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions_dpy as jf
from plotter import postplotter
from plotter import plot_acoeffs_datavsmodel as plot_acoeffs
import save_reduced_problem as SRP

#----------------------local directory structure------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
outdir = f"{scratch_dir}/qdpy_jax"
outdir_dpy = f"{scratch_dir}/dpy_jax"
plotdir = f"{scratch_dir}/plots"
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
# sigma2scale = np.load(f'{outdir}/sigma2scale.npy')
D_bsp_j_D_bsp_k = np.load(f'{outdir}/D_bsp_j_D_bsp_k.npy')
#------------------------------------------------------------------------# 
nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)
dim_hyper = int(np.loadtxt('.dimhyper'))
ellmax = np.max(ell0_arr)
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
len_s = len(sind_arr)
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
sparse_idx = jnp.asarray(sparse_idx)
ell0_arr = jnp.asarray(ell0_arr)
omega0_arr = jnp.asarray(omega0_arr)
aconv_denom = jnp.asarray(aconv_denom)


#------------------ defining functions ----------------------------
def print_info(itercount, tdiff, data_misfit, loss_diff,
               max_grads, model_misfit):
    print(f'[{itercount:3d} | ' +
          f'{tdiff:6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} ' +
          f'loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {max_grads:12.5e} ' +
          f'model_misfit={model_misfit:12.5e}')
    return None

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


def get_eigvalsfull(ipdata):
    eigval_model = np.array([])
    _hmatidx = np.moveaxis(sparse_idx, 1, -1)

    for mult_ind in range(nmults):
        _eigval_mult = np.zeros(2*ellmax+1)
        ell0 = ell0_arr[mult_ind]
        omegaref = omega0_arr[mult_ind]
        # pred_dense = sparse.bcoo_todense(ipdata[mult_ind],
        #                                  sparse_idx[mult_ind],
        #                                  shape=(dim_hyper,
        #                                         dim_hyper))
        pred_dense = sp_sparse.coo_matrix((ipdata[mult_ind],
                                           _hmatidx[mult_ind]),
                                          shape=(dim_hyper,
                                                 dim_hyper)).toarray()

        _eigval_mult[:2*ell0+1] = get_eigs(pred_dense)[:2*ell0+1]
        _eigval_mult = _eigval_mult/2./omegaref*GVARS.OM*1e6
        eigval_model = np.append(eigval_model,
                                 _eigval_mult)
    return eigval_model



def loop_in_mults(mult_ind, ipda):
    ipdata, ipdata_acoeff = ipda
    _eigval_mult = jnp.zeros(2*ellmax+1)
    ell0 = ell0_arr[mult_ind]
    omegaref = omega0_arr[mult_ind]
    pred_dense = sparse.bcoo_todense(ipdata[mult_ind],
                                     sparse_idx[mult_ind],
                                     shape=(dim_hyper, dim_hyper))
    _eigval_mult = get_eigs(pred_dense)[:2*ellmax+1]
    _eigval_mult = _eigval_mult/2./omegaref*GVARS.OM*1e6

    ipdata_acoeff = jdc_update(ipdata_acoeff,
                               ((Pjl[mult_ind] @ _eigval_mult)/
                                aconv_denom[mult_ind]),
                               (mult_ind * num_j,))
    ipda = (ipdata, ipdata_acoeff)
    return ipda


def model_Q(c_arr):
    pred = c_arr @ param_coeff_flat + fixed_part
    pred_acoeffs = jnp.zeros(num_j * nmults)
    __, pred_acoeffs = foril(0, nmults, loop_in_mults, (pred, pred_acoeffs))
    return pred_acoeffs


def data_misfit_arr_fn(c_arr):
    pred_acoeffs = model_Q(c_arr)
    data_misfit_arr = (data_acoeffs - pred_acoeffs)/acoeffs_sigma_HMI
    return data_misfit_arr


def data_misfit_fn(c_arr):
    data_misfit_arr = data_misfit_arr_fn(c_arr)
    return jnp.sum(jnp.square(data_misfit_arr))


def get_pc_pjl(c_arr):
    # function fori-loop
    def loop_in_c(cind, pc):
        __, pcpjl = foril(0, nmults, loop_in_mults, (pred, pc[cind]))
        pc = jidx_update(pc, jidx[cind, :], pcpjl)
        return pc

    pred = param_coeff_flat
    pc_pjl = jnp.zeros((len(c_arr), num_j * nmults))
    pc_pjl = foril(0, len(c_arr), loop_in_c, pc_pjl)
    return pc_pjl


def get_dhess_exact(c_arr):
    pc_pjl = get_pc_pjl(c_arr)
    gtg = pc_pjl @ (np.diag(1/acoeffs_sigma_HMI**2) @ pc_pjl.T)
    return 2*gtg


def get_GT_Cd_inv(c_arr):
    '''Function to compute G.T @ Cd_inv
    for obtaining the G^{-g} later.
    '''
    pc_pjl = get_pc_pjl(c_arr)
    GT_Cd_inv = pc_pjl @ np.diag(1/acoeffs_sigma_HMI**2)
    return GT_Cd_inv


def model_misfit_fn(c_arr, mu_scale=mu_scaling):
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
        # norm_c = jnp.square(cd - GVARS.ctrl_arr_dpt_full[sind_arr[i]])
        # cDc += jnp.sum(mu_depth * norm_c)
    return cDc


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


def hessian(f):
    return jacfwd(jacrev(f))
#---------------------- jitting the functions --------------------------#
data_hess_fn = hessian(data_misfit_fn)
model_hess_fn = hessian(model_misfit_fn)
grad_fn = jax.grad(loss_fn)
hess_fn = hessian(loss_fn)

_loss_fn = jit(loss_fn)
_grad_fn = jit(grad_fn)
_hess_fn = jit(hess_fn)
_update_H = jit(update_H)
_update_cgrad = jit(update_cgrad)

#---------------checking that the loaded data are correct----------------#
# adding the contribution from the fitting part
# and testing that arrays are very close
pred = fixed_part + true_params_flat @ param_coeff_flat
pred_eigvals = get_eigvalsfull(pred)
np.testing.assert_array_almost_equal(pred_eigvals, data)
print(f"[TESTING] pred_eigvals = data: PASSED")

#----------------------making the data_acoeffs---------------------------#
def loop_for_data(mult_ind, data_acoeff):
    ell0 = ell0_arr[mult_ind]
    data_omega = jdc(data, (mult_ind*(2*ellmax+1),), (2*ellmax+1,))
    Pjl_local = Pjl[mult_ind]
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_local @ data_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    return data_acoeff

data_acoeffs = jnp.zeros(num_j*nmults)
data_acoeffs = foril(0, nmults, loop_for_data, data_acoeffs)

#-------------checking that the acoeffs match correctly------------------#
pred_acoeffs = jnp.zeros(num_j * nmults)
__, pred_acoeffs = foril(0, nmults, loop_in_mults, (pred, pred_acoeffs))

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)
print(f"[TESTING] pred_acoeffs = data_acoeffs: PASSED")
#----------------------------------------------------------------------#
# changing to the HMI acoeffs if doing this for real data 
# data_acoeffs = GVARS.acoeffs_true
np.random.seed(3)
data_acoeffs_err = np.random.normal(loc=0, scale=acoeffs_sigma_HMI)
data_acoeffs = data_acoeffs + 0.0*data_acoeffs_err
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
#----------------------------------------------------------------------# 
len_data = len(data_acoeffs) #length of data
mu = PARGS.mu # regularization parameter

# changing the regularization as a function of depth
mu_depth = np.zeros_like(GVARS.ctrl_arr_dpt_full[0, :])
rth_soft = GVARS.rth + 0.01
width = 0.003
mu_depth = 0.5 * (1 - np.tanh((GVARS.knot_locs - rth_soft)/width))
mu_depth = 1e10 * jnp.sqrt(jnp.asarray(mu_depth)) / mu


#-----------------------the main training loop--------------------------#
# initialization of params
# c_init = np.random.uniform(5.0, 20.0, size=len(true_params_flat))*1e-4
# c_init += np.random.rand(len(c_init))
c_init = np.ones_like(true_params_flat)
c_init *= true_params_flat
print(f"Number of parameters = {len(c_init)}")

#------------------plotting the initial profiles-------------------#
c_arr_init_full = jf.c4fit_2_c4plot(GVARS, c_init,
                                    sind_arr, cind_arr)

# converting ctrl points to wsr and plotting
ctrl_zero_error = np.zeros_like(c_arr_init_full)
init_plot = postplotter.postplotter(GVARS, c_arr_init_full,
                                    ctrl_zero_error, 'init')
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
print(f"mu depth shape = {mu_depth.shape}")
print(f"ctrl full shape = {GVARS.ctrl_arr_dpt_full.shape}")
N = len(data_acoeffs)
loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-12
maxiter = 20
itercount = 0

hsuffix = f"{int(ARGS[4])}s.{GVARS.eigtype}.{GVARS.tslen}d.npy"
print(hsuffix)
if PARGS.read_hess:
    data_hess_dpy = np.load(f"{outdir_dpy}/dhess.{hsuffix}")
    model_hess_dpy = np.load(f"{outdir_dpy}/mhess.{hsuffix}")
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
print_info(itercount, tdiff, data_misfit,
           loss_diff, abs(grads).max(), model_misfit)
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
    print_info(itercount, t2-t1, data_misfit,
               loss_diff, abs(grads).max(), model_misfit)

    t2 = time.time()

t2s = time.time()
print(f"Total time taken = {(t2s-t1s):12.3f} seconds")
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
c_arr_fit = c_arr/true_params_flat

for i in range(len_s):
    print(c_arr_fit[i::len_s])

#-----------------finding the model covariance matrix------------------#
# can be shown that the model covariance matrix has the following form
# C_m = G^{-g} @ C_d @ G^{-g}.T
# G^{-g} = total_hess_inv @ G.T @ C_d_inv

# GT_Cd_inv = get_GT_Cd_inv(c_arr)
# G_g_inv = hess_inv @ GT_Cd_inv
# C_d = jnp.diag(acoeffs_sigma_HMI**2)
# C_m = jf.get_model_covariance(G_g_inv, C_d)
# ctrl_arr_err = jnp.sqrt(jnp.diag(C_m))
ctrl_arr_err = jnp.zeros_like(c_arr_fit)
#------------------plotting the post fitting profiles-------------------#
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                   sind_arr, cind_arr)

# making the full error array to pass into c4fit_2_c4plot
ctrl_arr_err_full = np.zeros_like(c_arr_fit_full)
ctrl_arr_err_full[:, -len(true_params_flat)//len_s:] =\
                            jnp.reshape(ctrl_arr_err, (3,-1), 'F')

c_arr_err_full = jnp.reshape(ctrl_arr_err_full, (3,-1), 'F')

# converting ctrl points to wsr and plotting
fit_plot = postplotter.postplotter(GVARS,
                                   c_arr_fit_full,
                                   c_arr_err_full, 'fit')

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
