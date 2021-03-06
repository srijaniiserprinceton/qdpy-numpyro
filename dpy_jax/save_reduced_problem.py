import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import norm
from scipy import integrate

NAX = np.newaxis
#-----------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
parser.add_argument("--smin", help="smin",
                    type=int, default=1)
parser.add_argument("--smax", help="smax",
                    type=int, default=5)
parser.add_argument("--batch_run", help="flag to indicate its a batch run",
                    type=int, default=0)
parser.add_argument("--batch_rundir", help="local directory for batch run",
                    type=str, default=".")
PARGS = parser.parse_args()
#------------------------------------------------------------------------# 
import jax.numpy as jnp
from jax.config import config
import jax
from jax import random
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril
from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)
# config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'gpu')
config.update('jax_enable_x64', True)

#-------------------------------------------------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

#----------------------import local packages------------------------#
from qdpy import build_hypermatrix_sparse as build_hm_sparse
from qdpy import globalvars as gvar_jax
from qdpy import jax_functions as jf
from plotter import plot_model_renorm as plot_renorm

if(not PARGS.batch_run):
    n0lminlmax_dir = f"{package_dir}/dpy_jax"
    outdir = f"{scratch_dir}/dpy_jax"
    from dpy_jax import sparse_precompute_acoeff as precompute
else:
    n0lminlmax_dir = f"{PARGS.batch_rundir}"
    outdir = f"{PARGS.batch_rundir}"
    sys.path.append(f"{PARGS.batch_rundir}")
    import sparse_precompute_acoeff_batch as precompute

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
                            numsplits=int(ARGS[8]),
                            smax_global=int(ARGS[9]))

#-------------------parameters to be inverted for--------------------#
# the indices of ctrl points that we want to invert for
ind_min, ind_max = 0, GVARS.ctrl_arr_dpt_clipped.shape[1]-1
cind_arr = np.arange(ind_min, ind_max+1)

# the angular degrees we want to invert for
smin, smax = PARGS.smin, PARGS.smax
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.arange(smin_ind, smax_ind+1)

#-----------------loading miscellaneous files--------------------------#
sfx = GVARS.filename_suffix
eigvals_model = jnp.asarray(np.load(f'{outdir}/eigvals_model.{sfx}.npy'))
eigvals_sigma_model = jnp.asarray(np.load(f'{outdir}/eigvals_sigma_model.{sfx}.npy'))
acoeffs_HMI = jnp.asarray(np.load(f'{outdir}/acoeffs_HMI.{sfx}.npy'))
acoeffs_sigma_HMI = jnp.asarray(np.load(f'{outdir}/acoeffs_sigma_HMI.{sfx}.npy'))
#----------------------------------------------------------------------#

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

#---------------computing miscellaneous shape parameters---------------# 
len_data = len(omega0_arr)
num_eigvals = len(eigvals_model)
nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

#---------section to add some more ctrl points to fixed part------------#
# array of ctrl points which aer non-zero for the fixed splines
c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
# filling in with the dpt wsr by default
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in range(smin_ind, smax_ind+1):
    for cind in cind_arr:
        c_fixed[sind, cind] = 0.0
        c_fixed[sind, cind] = 0.0

# scaling factor to get \delta\omega
fac_sig = 1./2./omega0_arr*GVARS.OM*1e6

# this is the fixed part of the diag
diag_evals_fixed = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                    fixed_hypmat_all_sparse,
                                                    c_fixed, nc, len_s).todense()
# the \delta\omega in muHz
diag_evals_fixed *= fac_sig

# we just need to save the noc_diag corresponding to the two ctrl_pts set to zero
noc_diag = []

for sind in range(smin_ind, smax_ind+1):
    noc_diag_s = []
    for cind in cind_arr:
        noc_diag_s.append(noc_hypmat_all_sparse[sind][cind].todense() * fac_sig)
    noc_diag.append(noc_diag_s)

#---------storing the true parameters which are allowed to vary-------#
true_params = np.zeros((smax_ind - smin_ind + 1, ind_max - ind_min + 1))

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        true_params[sind-smin_ind, ci] = GVARS.ctrl_arr_dpt_clipped[sind, cind]

#-------------saving the sigma in model params-------------------#
carr_sigma = np.zeros_like(true_params)

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        carr_sigma[sind-smin_ind, ci] = GVARS.ctrl_arr_sig_clipped[sind, cind]

#-------------computing the regularization terms-------------------#
# extracting the entire basis elements once for one s
# bsp_basis_one_s = precompute.get_bsp_basis_elements(GVARS.r)
bsp_basis_one_s = GVARS.bsp_basis

# retaining the ctrl points needed
bsp_basis_one_s = bsp_basis_one_s[cind_arr]

# making the bsp basis for all s according to the ordering of
# c_arr_flat
bsp_basis = np.zeros((len(sind_arr), len(cind_arr), len(GVARS.r)))

for sind in range(smin_ind, smax_ind+1):
    bsp_basis[sind-smin_ind] = bsp_basis_one_s

# flattening in the s and c dimension like ctrl_arr_flat
bsp_basis = np.reshape(bsp_basis, (len(sind_arr) * len(cind_arr), -1), 'F')
# np.save(f'{outdir}/bsp_basis.npy', bsp_basis)

# bsp_basis_full = np.load('bsp_basis_full.npy')
# acting the basis elements on with operator D
# D_bsp = jf.D(GVARS.bsp_basis, GVARS.r)
# start_idx = max(GVARS.knot_ind_th-4, 0)
D_bsp = jf.D(GVARS.bsp_basis_full, GVARS.r)

# calculating D_bsp_k * D_bsp_j and then integrating over radius
D_bsp_j_D_bsp_k_r = D_bsp[:, NAX, :] * D_bsp[NAX, :, :]
D_bsp_j_D_bsp_k = integrate.trapz(D_bsp_j_D_bsp_k_r, GVARS.r, axis=2)

#----------------------------------------------------------------------# 
# checking if the forward problem works with the above components
pred = diag_evals_fixed * 1.0

# flattening for easy dot product
true_params_flat = np.reshape(true_params,
                              (len(sind_arr) * len(cind_arr)), 'F')
carr_sigma_flat = np.reshape(carr_sigma,
                             (len(sind_arr) * len(cind_arr)), 'F')
noc_diag_flat = np.reshape(noc_diag,
                           (len(sind_arr) * len(cind_arr), -1), 'F')

print(len(sind_arr), len(cind_arr), len(sind_arr) * len(cind_arr))

# this is essentially the model function
pred += true_params_flat @ noc_diag_flat

# testing if the model still works after absorbing some ctrl points
# into the fixed part.
np.testing.assert_array_almost_equal(pred, eigvals_model)

#---------finding sigma of renorm params-------------------#
num_params = len(true_params_flat)
num_samples = int(1e6)
true_params_samples = np.zeros((num_params, num_samples))
true_params_flat_shaped = np.reshape(true_params_flat, (num_params, 1))

# looping over model params
for i in range(num_params):
    true_params_samples[i, :] = np.random.normal(loc=true_params_flat[i],
                                                 scale=np.abs(carr_sigma_flat[i]),
                                                 size=num_samples)
'''
# step 1 of renormalization
true_params_samples_renormed = jf.model_renorm(true_params_samples,
                                               true_params_flat_shaped,
                                               1.)

# array to store the sigma values to rescale renormed model params
sigma2scale = np.zeros(num_params)

for i in range(num_params):
    __, sigma2scale[i] = norm.fit(true_params_samples_renormed[i])
    # sigma2scale[i] = abs(carr_sigma_flat[i])
'''
#-------------saving miscellaneous files-------------------#
sfx = GVARS.filename_suffix

np.save(f'{outdir}/fixed_part.{sfx}.npy', diag_evals_fixed)
np.save(f'{outdir}/param_coeff_flat.{sfx}.npy', noc_diag_flat)
np.save(f'{outdir}/true_params_flat.{sfx}.npy', true_params_flat)
# np.save(f'{outdir}/model_params_sigma.npy', carr_sigma_flat*20.)
np.save(f'{outdir}/data_model.{sfx}.npy', eigvals_model)
np.save(f'{outdir}/cind_arr.{sfx}.npy', cind_arr)
np.save(f'{outdir}/sind_arr.{sfx}.npy', sind_arr)
np.save(f'{outdir}/D_bsp_j_D_bsp_k.{sfx}.npy', D_bsp_j_D_bsp_k)
print(f"--SAVING COMPLETE--")
