import os
import time
import argparse
from datetime import date
from datetime import datetime
from scipy import integrate
from tqdm import tqdm
#------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument("--mu", help="regularization",
                    type=float, default=0.)
parser.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
parser.add_argument("--synth", help="use synthetic data",
                    type=bool, default=False)
parser.add_argument("--noise", help="add noise",
                    type=bool, default=True)
parser.add_argument("--rundir", help="local directory for batch run",
                    type=str, default=".")
parser.add_argument("--mu_batchdir", help="directory of converged mu",
                    type=str, default=".")
PARGS = parser.parse_args()
#------------------------------------------------------------------------# 
from collections import namedtuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
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
from jax.experimental import sparse
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
print(jax.devices())
#------------------------------------------------------------------------# 
from qdpy import globalvars as gvar_jax
from qdpy import jax_functions as jf
#------------------------------------------------------------------------# 
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

n0lminlmax_dir_dpy = f"{PARGS.rundir}/dpy_files"
n0lminlmax_dir_qdpy = f"{PARGS.rundir}/qdpy_files"
outdir = f"{PARGS.rundir}"
summdir = f"{PARGS.rundir}/summaryfiles"
plotdir = f"{PARGS.rundir}/plots"

# defining the directories for dpy_jax and qdpy_jax files
dpy_dir = f"{PARGS.rundir}/dpy_files"
dpyfull_dir = f"{PARGS.rundir}/dpy_full_hess"
qdpy_dir = f"{PARGS.rundir}/qdpy_files"
#------------------------------------------------------------------------# 
sys.path.append(f"{package_dir}/plotter")
import postplotter
import plot_acoeffs_datavsmodel as plot_acoeffs
#------------------------------------------------------------------------# 
# summary dictionary where all results will be stored
soln_summary = {}
soln_summary['params'] = {}
soln_summary['params']['qdpy'] = {}
soln_summary['params']['dpy'] = {}
#------------------------------------------------------------------------# 
ARGS_D = np.loadtxt(f"{n0lminlmax_dir_dpy}/.n0-lmin-lmax.dat")
GVARS_D = gvar_jax.GlobalVars(n0=int(ARGS_D[0]),
                              lmin=int(ARGS_D[1]),
                              lmax=int(ARGS_D[2]),
                              rth=ARGS_D[3],
                              knot_num=int(ARGS_D[4]),
                              load_from_file=int(ARGS_D[5]),
                              relpath=n0lminlmax_dir_dpy,
                              instrument=PARGS.instrument,
                              tslen=int(ARGS_D[6]),
                              daynum=int(ARGS_D[7]),
                              numsplits=int(ARGS_D[8]),
                              smax_global=int(ARGS_D[9]))

soln_summary['params']['dpy']['n0'] = int(ARGS_D[0])
soln_summary['params']['dpy']['lmin'] = int(ARGS_D[1])
soln_summary['params']['dpy']['lmax'] = int(ARGS_D[2])
soln_summary['params']['dpy']['rth'] = ARGS_D[3]
soln_summary['params']['dpy']['knot_num'] = int(ARGS_D[4])
soln_summary['params']['dpy']['GVARS'] = jf.dict2obj(GVARS_D.__dict__)
#------------------------------------------------------------------------# 
ARGS_Q = np.loadtxt(f"{n0lminlmax_dir_qdpy}/.n0-lmin-lmax.dat")
GVARS_Q = gvar_jax.GlobalVars(n0=int(ARGS_Q[0]),
                              lmin=int(ARGS_Q[1]),
                              lmax=int(ARGS_Q[2]),
                              rth=ARGS_Q[3],
                              knot_num=int(ARGS_Q[4]),
                              load_from_file=int(ARGS_Q[5]),
                              relpath=n0lminlmax_dir_qdpy,
                              instrument=PARGS.instrument,
                              tslen=int(ARGS_Q[6]),
                              daynum=int(ARGS_Q[7]),
                              numsplits=int(ARGS_Q[8]),
                              smax_global=int(ARGS_Q[9]))


soln_summary['params']['qdpy']['n0'] = int(ARGS_Q[0])
soln_summary['params']['qdpy']['lmin'] = int(ARGS_Q[1])
soln_summary['params']['qdpy']['lmax'] = int(ARGS_Q[2])
soln_summary['params']['qdpy']['rth'] = ARGS_Q[3]
soln_summary['params']['qdpy']['knot_num'] = int(ARGS_Q[4])
soln_summary['params']['qdpy']['GVARS'] = jf.dict2obj(GVARS_Q.__dict__)

#-------------loading precomputed files for the problem-------------------# 
sfxD = GVARS_D.filename_suffix
data_D = np.load(f'{dpy_dir}/data_model.{sfxD}.npy')
true_params_flat_D = np.load(f'{dpy_dir}/true_params_flat.{sfxD}.npy')
param_coeff_flat_D = np.load(f'{dpy_dir}/param_coeff_flat.{sfxD}.npy')
fixed_part_D = np.load(f'{dpy_dir}/fixed_part.{sfxD}.npy')
acoeffs_sigma_HMI_D = np.load(f'{dpy_dir}/acoeffs_sigma_HMI.{sfxD}.npy')
acoeffs_HMI_D = np.load(f'{dpy_dir}/acoeffs_HMI.{sfxD}.npy')
cind_arr_D = np.load(f'{dpy_dir}/cind_arr.{sfxD}.npy')
sind_arr_D = np.load(f'{dpy_dir}/sind_arr.{sfxD}.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly_D = np.load(f'{dpy_dir}/RL_poly.{sfxD}.npy')
sigma2scale = np.load(f'{dpy_dir}/sigma2scale.{sfxD}.npy')
D_bsp_j_D_bsp_k = np.load(f'{dpy_dir}/D_bsp_j_D_bsp_k.{sfxD}.npy')

#-------------loading precomputed files for the problem-------------------#
sfxQ = GVARS_Q.filename_suffix
data_Q = np.load(f'{qdpy_dir}/data_model.{sfxQ}.npy')
true_params_flat_Q = np.load(f'{qdpy_dir}/true_params_flat.{sfxQ}.npy')
param_coeff_flat_Q = np.load(f'{qdpy_dir}/param_coeff_flat.{sfxQ}.npy')
fixed_part_Q = np.load(f'{qdpy_dir}/fixed_part.{sfxQ}.npy')
sparse_idx_Q = np.load(f'{qdpy_dir}/sparse_idx.{sfxQ}.npy')
acoeffs_sigma_HMI_Q = np.load(f'{qdpy_dir}/acoeffs_sigma_HMI.{sfxQ}.npy')
acoeffs_HMI_Q = np.load(f'{qdpy_dir}/acoeffs_HMI.{sfxQ}.npy')
cind_arr_Q = np.load(f'{qdpy_dir}/cind_arr.{sfxQ}.npy')
sind_arr_Q = np.load(f'{qdpy_dir}/sind_arr.{sfxQ}.npy')
ell0_arr_Q = np.load(f'{qdpy_dir}/ell0_arr.{sfxQ}.npy')
omega0_arr_Q = np.load(f'{qdpy_dir}/omega0_arr.{sfxQ}.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
RL_poly_Q = np.load(f'{qdpy_dir}/RL_poly.{sfxQ}.npy')

#-------------------Miscellaneous parameters---------------------------#
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

try:
    knee_mu = []
    if(int(ARGS_D[9]) == int(ARGS_Q[9])):
        smax_global = int(ARGS_D[9])
    else:
        print("DPT and QDPT using different smax_global")
        print(f"DPY-dir = {n0lminlmax_dir_dpy}")
        print(f"QDPY-dir = {n0lminlmax_dir_qdpy}")
        sys.exit()
        
    for s in range(1, smax_global+1, 2):
        try:
            knee_mu.append(np.load(f"{PARGS.mu_batchdir}/muval.s{s}.npy"))
        except FileNotFoundError:
            knee_mu.append(1.0)
    knee_mu = np.asarray(knee_mu)
    knee_mu *= 1.
    print('Using optimal mu.')
except FileNotFoundError:
    knee_mu = np.array([1.e-4, 1.e-4, 5.e-4])
    print('Not using optimal mu.')

print(f"knee_mu = {knee_mu}")
wsr_sigma = np.load(f"{dpyfull_dir}/wsr_sigma.npy")
#-----------------------------------------------------------------------# 
nmults_Q = len(GVARS_Q.ell0_arr)
num_j_Q = len(GVARS_Q.s_arr)
dim_hyper_Q = int(np.loadtxt(f'{qdpy_dir}/.dimhyper'))
ellmax_Q = np.max(ell0_arr_Q)
smin_Q = min(GVARS_Q.s_arr)
smax_Q = max(GVARS_Q.s_arr)

# slicing the Pjl correctly in angular degree s 
Pjl_Q = RL_poly_Q[:, smin_Q:smax_Q+1:2, :]

#------------------setting common DPT and QDPT params-----------------#
np.testing.assert_array_almost_equal(GVARS_D.s_arr, GVARS_Q.s_arr)
np.testing.assert_array_almost_equal(cind_arr_D, cind_arr_Q)
np.testing.assert_array_almost_equal(sind_arr_D, sind_arr_Q)
np.testing.assert_array_almost_equal(true_params_flat_D,
                                     true_params_flat_Q)
len_s = len(sind_arr_D)
print(f"len_s = {len_s}")
true_params_flat = true_params_flat_D
num_j = num_j_D

del num_j_D, num_j_Q
del true_params_flat_D, true_params_flat_Q

# the regularizing parameter
mu = PARGS.mu

# calculating the DPT & model hessian once since it doesn't depend on c_arr
suffix = f"{int(ARGS_Q[4])}s.{GVARS_Q.eigtype}.{sfxQ}"
data_hess_dpy = np.load(f"{dpyfull_dir}/dhess.{suffix}.npy")
model_hess_dpy = np.load(f"{dpyfull_dir}/mhess.{suffix}.npy")
true_params_iter = np.load(f"{dpyfull_dir}/carr_fit_{mu:.5e}.npy")
hess_inv = jnp.linalg.inv(data_hess_dpy + mu * model_hess_dpy)
#------------------------------------------------------------------------#
# calculating the denominator of DPT a-coefficient converion apriori
# shape (nmults_D, num_j_D)
aconv_denom_D = np.zeros((nmults_D, Pjl_D.shape[1]))
for mult_ind in range(nmults_D):
    aconv_denom_D[mult_ind] = np.diag(Pjl_D[mult_ind] @ Pjl_D[mult_ind].T)

#------------------------------------------------------------------------#
# calculating the denominator of QDPT a-coefficient converion apriori   
# shape (nmults_Q, num_j_Q)      
aconv_denom_Q = np.zeros((nmults_Q, Pjl_Q.shape[1]))
for mult_ind in range(nmults_Q):
    aconv_denom_Q[mult_ind] = np.diag(Pjl_Q[mult_ind] @ Pjl_Q[mult_ind].T)
#-------------------------converting to device array---------------------# 
Pjl_D = jnp.asarray(Pjl_D)
data_D = jnp.asarray(data_D)
param_coeff_flat_D = jnp.asarray(param_coeff_flat_D)
fixed_part_D = jnp.asarray(fixed_part_D)
acoeffs_HMI_D = jnp.asarray(acoeffs_HMI_D)
acoeffs_sigma_HMI_D = jnp.asarray(acoeffs_sigma_HMI_D)
aconv_denom_D = jnp.asarray(aconv_denom_D)

#-------------------------converting to device array---------------------#
Pjl_Q = jnp.asarray(Pjl_Q)
data_Q = jnp.asarray(data_Q)
param_coeff_flat_Q = jnp.asarray(param_coeff_flat_Q)
fixed_part_Q = jnp.asarray(fixed_part_Q)
acoeffs_sigma_HMI_Q = jnp.asarray(acoeffs_sigma_HMI_Q)
sparse_idx_Q = jnp.asarray(sparse_idx_Q)
ell0_arr_Q = jnp.asarray(ell0_arr_Q)
omega0_arr_Q = jnp.asarray(omega0_arr_Q)
aconv_denom_Q = jnp.asarray(aconv_denom_Q)

true_params_flat = jnp.asarray(true_params_flat)

#----------------------making the data_acoeffs---------------------------# 
def loop_in_mults_D(mult_ind, data_acoeff):
    data_omega = jdc(data_D, (mult_ind*dim_hyper_D,), (dim_hyper_D,))
    data_acoeff = jdc_update(data_acoeff,
                             ((Pjl_D[mult_ind] @ data_omega)/
                              aconv_denom_D[mult_ind]),
                             (mult_ind * num_j,))
    return data_acoeff


def loop_in_mults_Q(mult_ind, data_acoeff):
    data_omega = jdc(data_Q, (mult_ind*(2*ellmax_Q+1),), (2*ellmax_Q+1,))
    data_acoeff = jdc_update(data_acoeff,
                             ((Pjl_Q[mult_ind] @ data_omega)/
                              aconv_denom_Q[mult_ind]),
                             (mult_ind * num_j,))
    return data_acoeff


data_acoeffs_D = jnp.zeros(num_j*nmults_D)
data_acoeffs_Q = jnp.zeros(num_j*nmults_Q)
data_acoeffs_D = foril(0, nmults_D, loop_in_mults_D, data_acoeffs_D)
data_acoeffs_Q = foril(0, nmults_Q, loop_in_mults_Q, data_acoeffs_Q)

len_data = len(data_acoeffs_D) + len(data_acoeffs_Q)
#--------------------------------------------------------------------------# 
def data_misfit_fn_D(c_arr, dac_D, fullfac):
    # predicted DPT a-coefficients
    pred_acoeffs_D = model_D(c_arr, fullfac)
    data_misfit_arr_D = (pred_acoeffs_D - dac_D)/acoeffs_sigma_HMI_D
    dm = 0.0
    for i in range(len_s):
        dm += jnp.sum(jnp.square(data_misfit_arr_D[i::len_s]))
    return dm
    """
    dmarr = [jnp.asarray(data_misfit_arr_D[i::len_s]) for i in range(len_s)]
    
    def loop_dm(i, dm):
        dm =  jdc_update(dm, dm[0] + jnp.sum(jnp.square(dmarr[i])), (0,))
        return dm
    
    dm = jnp.array([0.0, 0.0])
    return foril(0, len_s, loop_dm, dm)[0]
    """


def data_misfit_fn_Q(c_arr, dac_Q, fullfac):
    # predicted QDPT a-coefficients
    pred_acoeffs_Q = model_Q(c_arr, fullfac)
    data_misfit_arr_Q = (pred_acoeffs_Q - dac_Q)/acoeffs_sigma_HMI_Q
    """
    dmarr = [jnp.asarray(data_misfit_arr_Q[i::len_s]) for i in range(len_s)]
    
    def loop_dm(i, dm):
        dm =  jdc_update(dm, dm[0] + jnp.sum(jnp.square(dmarr[i])), (0,))
        return dm
    
    dm = jnp.array([0.0, 0.0])
    return foril(0, len_s, loop_dm, dm)[0]
    """
    dm = 0.0
    for i in range(len_s):
        dm += jnp.sum(jnp.square(data_misfit_arr_Q[i::len_s]))
    return dm


def model_misfit_fn(c_arr, fullfac, mu_scale=knee_mu):
    # Djk is the same for s=1, 3, 5
    Djk = D_bsp_j_D_bsp_k
    sidx, eidx = 0, GVARS_D.knot_ind_th
    cDc = 0.0

    for i in range(len_s):
        carr_padding = GVARS_D.ctrl_arr_dpt_full[sind_arr_D[i], sidx:eidx] * fullfac
        cd = jnp.append(carr_padding, c_arr[i::len_s])
        lambda_factor = jnp.trace(data_hess_dpy[i::len_s, i::len_s])/\
                        (2 * jnp.trace(Djk[-eidx:,-eidx:]))
        lambda_factor *= len_data
        cDc += mu_scale[i] * cd @ Djk @ cd * lambda_factor
    return cDc


def hessian_D(f):
    return jacfwd(jacrev(f))


def hessian_Q(f):
    return jacfwd(jacfwd(f))


def loss_fn(c_arr, dac_D, dac_Q, fullfac):
    data_misfit_val_D = data_misfit_fn_D(c_arr, dac_D, fullfac)
    data_misfit_val_Q = data_misfit_fn_Q(c_arr, dac_Q, fullfac)
    model_misfit_val = model_misfit_fn(c_arr, fullfac)

    misfit = (data_misfit_val_D +
              data_misfit_val_Q +
              mu*model_misfit_val)
    return misfit


def update_H(c_arr, grads, hess_inv):
    return jax.tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)


def get_wsr(carr):
    carr1 = GVARS_D.ctrl_arr_dpt_full * 1.0
    for i in range(len_s):
        carr1[sind_arr_D[i], GVARS_D.knot_ind_th:] = carr[i]
    wsr_full = carr1 @ GVARS_D.bsp_basis_full
    wsr = [wsr_full[sind_arr_D[i]] for i in range(len_s)]
    return np.array(wsr)


def compute_misfit_wsr(arr1, arr2, sig):
    wsr1 = get_wsr(arr1)
    wsr2 = get_wsr(arr2)
    absdiff_by_sig = abs(wsr1 - wsr2)/sig
    diffsig_s = [absdiff_by_sig[i] for i in range(len_s)]
    return [max(diffsig_s[i]) for i in range(len_s)]

def compute_misfit_wsr_2(arr1, arr2, sig):
    wsr1 = get_wsr(arr1)
    wsr2 = get_wsr(arr2)
    diff_by_sig = (wsr1 - wsr2)/sig

    integrand = diff_by_sig**2
    integral = integrate.trapz(integrand, GVARS_D.r, axis=1)

    sum_integral_alls = np.sum(integral)
    sum_integral_alls_scaled = sum_integral_alls / (GVARS_D.r.max() - GVARS_D.rth)
    return sum_integral_alls_scaled
#----------------------------------------------------------------------# 
# the DPT model function that returns a-coefficients
def model_D(c_arr, fullfac):
    """fullfac is used to either include or exclude the fixed part"""
    pred_acoeffs = jnp.zeros(num_j * nmults_D)
    pred = fixed_part_D*fullfac + c_arr @ param_coeff_flat_D

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper_D,), (dim_hyper_D,))
        pred_acoeff = jdc_update(pred_acoeff,
                                 ((Pjl_D[mult_ind] @ pred_omega)/
                                  aconv_denom_D[mult_ind]),
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults_D, loop_in_mults, pred_acoeffs)
    return pred_acoeffs


# model_D_ = jit(model_D)
pred_acoeffs_D = model_D(true_params_flat, 1)

dac_Q = data_acoeffs_Q * 1.0
dac_D = data_acoeffs_D * 1.0
# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs_D, data_acoeffs_D)
print(f"[TESTING] pred_acoeffs_D = data_acoeffs_D: PASSED -- " +
      f"maxdiff = {abs(pred_acoeffs_D - data_acoeffs_D).max():.5e}; " + 
      f"misfit = {data_misfit_fn_D(true_params_flat, dac_D, 1):.5e}")
#----------------------------------------------------------------------#
# the QDPT model function that returns a-coefficients 
def model_Q(c_arr, fullfac):
    pred_acoeffs = jnp.zeros(num_j * nmults_Q)
    pred = c_arr @ param_coeff_flat_Q + fixed_part_Q * fullfac

    def loop_in_mults(mult_ind, pred_acoeff):
        _eigval_mult = jnp.zeros(2*ellmax_Q+1)
        ell0 = ell0_arr_Q[mult_ind]
        omegaref = omega0_arr_Q[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind],
                                         sparse_idx_Q[mult_ind],
                                         shape=(dim_hyper_Q, dim_hyper_Q))
        _eigval_mult = get_eigs(pred_dense)[:2*ellmax_Q+1]
        _eigval_mult = _eigval_mult/2./omegaref*GVARS_Q.OM*1e6

        pred_acoeff = jdc_update(pred_acoeff,
                                 ((Pjl_Q[mult_ind] @ _eigval_mult)/
                                  aconv_denom_Q[mult_ind]),
                                 (mult_ind * num_j,))
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

# model_Q_ = jit(model_Q)
pred_acoeffs_Q = model_Q(true_params_flat, 1)

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs_Q, data_acoeffs_Q)
print(f"[TESTING] pred_acoeffs_Q = data_acoeffs_Q: PASSED -- " +
      f"maxdiff = {abs(pred_acoeffs_Q - data_acoeffs_Q).max():.5e}; " +
      f"misfit = {data_misfit_fn_Q(true_params_flat, dac_Q, 1):.5e}")
print(f"[TESTING] total-loss = {loss_fn(true_params_flat, dac_D, dac_Q, 1):.5e}")
#----------------------------------------------------------------------#
# synthetic or real data; if synthetic whether to add artifical noise 
if PARGS.synth:
    print("Using synthetic data")
    if PARGS.noise:
        print("--Using gaussian noise")
        np.random.seed(3)
        data_acoeffs_err_D = np.random.normal(loc=0, scale=acoeffs_sigma_HMI_D)
        data_acoeffs_D = data_acoeffs_D + 0.0*data_acoeffs_err_D
        
        np.random.seed(3)
        data_acoeffs_err_Q = np.random.normal(loc=0, scale=acoeffs_sigma_HMI_Q)
        data_acoeffs_Q = data_acoeffs_Q + 0.0*data_acoeffs_err_Q

else:
    print("Using observed data")
    data_acoeffs_D = GVARS_D.acoeffs_true
    data_acoeffs_Q = GVARS_Q.acoeffs_true

# making the pred_data corresponding to the true_params_iter 
# true_params_flat = jnp.asarray(true_params_iter) * 1.0
'''
pred_acoeffs_D = model_D_(true_params_flat, 1)
pred_acoeffs_Q = model_Q_(true_params_flat, 1)
'''

print(f"data_acoeffs_D_flat = {data_acoeffs_D[:15]}")
print(f"data_acoeffs_Q_flat = {data_acoeffs_Q[:15]}")
# pred_acoeffs_D = model_D_(true_params_iter, 1)                                              
# pred_acoeffs_Q = model_Q_(true_params_iter, 1)
# print(f"data_acoeffs_D_iter = {pred_acoeffs_D[:15]}")
# print(f"data_acoeffs_Q_iter = {pred_acoeffs_Q[:15]}")


data_acoeffs_out_HMI_D = GVARS_D.acoeffs_out_HMI
data_acoeffs_out_HMI_Q = GVARS_Q.acoeffs_out_HMI


# using real data
data_acoeffs_Q = GVARS_Q.acoeffs_true
data_acoeffs_D = GVARS_D.acoeffs_true
'''
# using synthetic data without noise from the true_params_iter model
data_acoeffs_Q = pred_acoeffs_Q
data_acoeffs_D = pred_acoeffs_D
'''

'''
#----------------------------------------------------------------------# 
# plotting acoeffs pred and data to see if we should expect good fit
plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs_D, data_acoeffs_D,
                                      data_acoeffs_out_HMI_D,
                                      acoeffs_sigma_HMI_D, 'ref_D',
                                      plotdir=plotdir)

plot_acoeffs.plot_acoeffs_datavsmodel(pred_acoeffs_Q, data_acoeffs_Q,
                                      data_acoeffs_out_HMI_Q,
                                      acoeffs_sigma_HMI_Q, 'ref_Q',
                                      plotdir=plotdir)
'''
#---------------------- jitting the functions --------------------------#
grad_fn = jax.grad(loss_fn)
_grad_fn = grad_fn #jit(grad_fn)
_update_H = update_H #jit(update_H)
_loss_fn = loss_fn #jit(loss_fn)
#-----------------------initialization of params------------------#
c_init = np.ones_like(true_params_flat)
#------------------plotting the initial profiles-------------------#
c_arr_init_full = jf.c4fit_2_c4plot(GVARS_D, c_init*true_params_flat,
                                    sind_arr_D, cind_arr_D)
ctrl_zero_err = np.zeros_like(c_arr_init_full)

# converting ctrl points to wsr and plotting
init_plot = postplotter.postplotter(GVARS_D, c_arr_init_full,
                                    ctrl_zero_err, 'init',
                                    plotdir=plotdir)

#------------------plotting the DPT iterative profiles-------------------#
_tpiter_full = jf.c4fit_2_c4plot(GVARS_D, true_params_flat, sind_arr_D, cind_arr_D)
fit_plot = postplotter.postplotter(GVARS_D,
                                   _tpiter_full,
                                   ctrl_zero_err,
                                   f'fit-dpyiter',
                                   plotdir=plotdir)
#------------------------------------------------------------------------
# getting the renormalized model parameters
c_arr = c_init * true_params_flat
#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
init_acoeffs_D = model_D(c_arr, 1)
init_acoeffs_Q = model_Q(c_arr, 1)

"""
plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs_D, data_acoeffs_D,
                                      data_acoeffs_out_HMI_D,
                                      acoeffs_sigma_HMI_D, 'init_D',
                                      plotdir=plotdir, len_s=len_s)

plot_acoeffs.plot_acoeffs_datavsmodel(init_acoeffs_Q, data_acoeffs_Q,
                                      data_acoeffs_out_HMI_Q,
                                      acoeffs_sigma_HMI_Q, 'init_Q',
                                      plotdir=plotdir, len_s=len_s)

plot_acoeffs.plot_acoeffs_dm_scaled(init_acoeffs_D, data_acoeffs_D,
                                    data_acoeffs_out_HMI_D,
                                    acoeffs_sigma_HMI_D, 'init_D',
                                    plotdir=plotdir, len_s=len_s)

plot_acoeffs.plot_acoeffs_dm_scaled(init_acoeffs_Q, data_acoeffs_Q,
                                    data_acoeffs_out_HMI_Q,
                                    acoeffs_sigma_HMI_Q, 'init_Q',
                                    plotdir=plotdir, len_s=len_s)
"""
#----------------------------------------------------------------------#

loss = 1e25
loss_diff = loss - 1.
loss_arr = []
loss_threshold = 1e-1
kmax = 20
N0 = 1
itercount = 0

model_misfit = model_misfit_fn(true_params_flat, 1)
loss = _loss_fn(true_params_flat, data_acoeffs_D, data_acoeffs_Q, 1)
grads = _grad_fn(true_params_flat, data_acoeffs_D, data_acoeffs_Q, 1)
data_misfit = loss - mu * model_misfit
tinit = 0
print(f'[{itercount:3d} | {tinit:6.1f} sec ] ' +
      f'data_misfit = {data_misfit:12.5e} loss-diff = {loss_diff:12.5e}; ' +
      f'max-grads = {abs(grads).max():12.5e} model_misfit={model_misfit:12.5e}')

carr_total = 0.
carr_total += true_params_flat
c_arr_allk = [carr_total]
int_k = []
kiter = 0

dm_list = []
mm_list = []
dm_list.append(data_misfit)
mm_list.append(model_misfit)

t1s = time.time()
# while ((abs(loss_diff) > loss_threshold) and
#        (itercount < maxiter)):
while(kiter < kmax):
    # for ii in tqdm(range(2**kiter * N0), desc=f"k={kiter}"):
    t1 = time.time()
    loss_prev = loss
    grads = _grad_fn(carr_total, data_acoeffs_D, data_acoeffs_Q, 1)
    carr_total = _update_H(carr_total, grads, hess_inv)
    loss = _loss_fn(carr_total, data_acoeffs_D, data_acoeffs_Q, 1)
    
    model_misfit = model_misfit_fn(carr_total, 1)
    data_misfit = loss - mu*model_misfit
    loss_diff = loss_prev - loss
    
    dm_list.append(data_misfit)
    mm_list.append(model_misfit)
    
    loss_arr.append(loss)
    
    itercount += 1
    t2 = time.time()
    
    print(f'[{itercount:3d} | {(t2-t1):6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {abs(grads).max():12.5e} ' +
          f'model_misfit={model_misfit:12.5e}')

    c_arr_allk.append(carr_total)
    int_k.append(compute_misfit_wsr_2(c_arr_allk[-1], c_arr_allk[-2], wsr_sigma))
    '''
    print(f"  [{kiter}] --- int diff = {int_k[-1]}")
    if kiter > 1:
        if int_k[-1] > int_k[-2]:
            carr_total = c_arr_allk[-2]
            break
    '''
    #------------------plotting the post fitting profiles-------------------#
    _citer_full = jf.c4fit_2_c4plot(GVARS_D, carr_total, sind_arr_D, cind_arr_D)
    fit_plot = postplotter.postplotter(GVARS_D,
                                       _citer_full,
                                       ctrl_zero_err,
                                       f'fit-hyb-kiter-{kiter}',
                                       plotdir=plotdir)

    #------------------------------------------------------------------------
    kiter += 1

t2s = time.time()
t21s_min = (t2s - t1s)/60.
print(f"Total time taken = {t21s_min:7.2f} minutes")

data_misfit_val_D = data_misfit_fn_D(carr_total, data_acoeffs_D, 1)
data_misfit_val_Q = data_misfit_fn_Q(carr_total, data_acoeffs_Q, 1)
total_misfit = data_misfit_val_D + data_misfit_val_Q
num_data = len(data_acoeffs_D) + len(data_acoeffs_Q)
rms = np.sqrt(total_misfit/num_data)
print(f"chisq = {total_misfit:.5f}")
print(f"rms = {rms:.5f}")

#----------------------------------------------------------------------#
# plotting acoeffs from initial data and HMI data
final_acoeffs_D = model_D(carr_total, 1)
final_acoeffs_Q = model_Q(carr_total, 1)

"""
plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs_D, data_acoeffs_D,
                                      data_acoeffs_out_HMI_D,
                                      acoeffs_sigma_HMI_D, 'final_D',
                                      plotdir=plotdir, len_s=len_s)

plot_acoeffs.plot_acoeffs_datavsmodel(final_acoeffs_Q, data_acoeffs_Q,
                                      data_acoeffs_out_HMI_Q,
                                      acoeffs_sigma_HMI_Q, 'final_Q',
                                      plotdir=plotdir, len_s=len_s)
"""
#----------------------------------------------------------------------# 
# reconverting back to model_params in units of true_params_flat
c_arr_fit = carr_total/true_params_flat
for i in range(len_s):
    print(f"------------- s = {GVARS_D.s_arr[sind_arr_D[i]]} -----------------------")
    print(c_arr_fit[i::len_s])

#------------------plotting the post fitting profiles-------------------#
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS_D, c_arr_fit*true_params_flat,
                                   sind_arr_D, cind_arr_D)

# converting ctrl points to wsr and plotting
fit_plot = postplotter.postplotter(GVARS_D, c_arr_fit_full,
                                   ctrl_zero_err, 'fit',
                                   plotdir=plotdir)

#------------------------------------------------------------------------
soln_summary['c_arr_fit'] = c_arr_fit
soln_summary['true_params_flat'] = true_params_flat
soln_summary['cind_arr'] = cind_arr_D
soln_summary['sind_arr'] = sind_arr_D
soln_summary['data-misfit'] = dm_list
soln_summary['model-misfit'] = mm_list
soln_summary['c_arr_allk'] = c_arr_allk

soln_summary['acoeff'] = {}
soln_summary['acoeff']['fit_D'] = final_acoeffs_D
soln_summary['acoeff']['fit_Q'] = final_acoeffs_Q
soln_summary['acoeff']['data_D'] = data_acoeffs_D
soln_summary['acoeff']['data_Q'] = data_acoeffs_Q
soln_summary['acoeff']['sigma_D'] = acoeffs_sigma_HMI_D
soln_summary['acoeff']['sigma_Q'] = acoeffs_sigma_HMI_Q

soln_summary['data_hess'] = data_hess_dpy
soln_summary['model_hess'] = model_hess_dpy
soln_summary['loss_arr'] = loss_arr
soln_summary['mu'] = mu
soln_summary['knee_mu'] = knee_mu
soln_summary['chisq'] = total_misfit
soln_summary['rms'] = rms

todays_date = date.today()
timeprefix = datetime.now().strftime("%H.%M")
dateprefix = f"{todays_date.day:02d}.{todays_date.month:02d}.{todays_date.year:04d}"
fsuffix = f"{dateprefix}-{timeprefix}-{suffix}"
jf.save_obj(soln_summary, f"{summdir}/summary-{fsuffix}")
