import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

import jax.numpy as jnp
from jax.config import config
import jax
from jax import random
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril
from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)
config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'gpu')
config.update('jax_enable_x64', True)

#----------------------import custom packages------------------------#
from qdpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions_dpy as jf
from dpy_jax import sparse_precompute_acoeff as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)

#-------------------parameters to be inverted for--------------------#
# the indices of ctrl points that we want to invert for
ind_min, ind_max = 0, 40#0, 3
cind_arr = np.arange(ind_min, ind_max + 1)

# the angular degrees we want to invert for
smin, smax = 3, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.arange(smin_ind, smax_ind+1)
#---------------------------------------------------------------------#

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

#-----------------loading miscellaneous files--------------------------#
eigvals_model = jnp.asarray(np.load('eigvals_model.npy'))
eigvals_sigma_model = jnp.asarray(np.load('eigvals_sigma_model.npy'))
acoeffs_HMI = jnp.asarray(np.load('acoeffs_HMI.npy'))
acoeffs_sigma_HMI = jnp.asarray(np.load('acoeffs_sigma_HMI.npy'))
#----------------------------------------------------------------------#

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

#---------------computing miscellaneous shape parameters---------------# 
len_data = len(omega0_arr)
num_eigvals = len(eigvals_model)
nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)
#----------------------------------------------------------------------# 
'''
def get_posterior_grid(cind, sind, N):
    # array containing the factors for the param space
    fac = jnp.linspace(0.01, 2.0, N)
    
    # misfit as a function of scaling param
    misfit_model_arr = jnp.zeros(N)
    
    # looping over factors
    def loop_facind(facind, misfits):
        misfit_mod_arr = misfits
        cdpt = GVARS.ctrl_arr_dpt_clipped*1.0
        cdpt = jidx_update(cdpt, jidx[sind, cind],
                           ctrl_arr_dpt[sind, cind] * fac[facind])

        diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  cdpt, nc, len_s)
        delta_omega = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6
        delta_omega_model = delta_omega - eigvals_model

        # delta_omega_model /= eigvals_sigma
        misfit_mod = -0.5*np.sum(delta_omega_model**2)/len_data
        misfit_mod_arr = jidx_update(misfit_mod_arr, jidx[facind], misfit_mod)

        return misfit_mod_arr
    
    return fac, foril(0, N, loop_facind, misfit_model_arr)

# jitting the function
_get_posterior_grid = jax.jit(get_posterior_grid, static_argnums=(2,))

N = 100

# the misfit array of shape (s x ctrl_pts x N)
misfit_arr_all = jnp.zeros((smax_ind - smin_ind + 1, len(cind_arr), N))

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        fac, misfit_cs = _get_posterior_grid(cind, sind, N)

        misfit_arr_all = jidx_update(misfit_arr_all,
                                     jidx[sind-1, ci, :],
                                     misfit_cs)

# 1D plot of the misfit as we scan across the terrain

fig, axs = plt.subplots(smax_ind-smin_ind+1,
                        ind_max-ind_min+1,
                        figsize=(12, 8), sharex=True)
axs = np.reshape(axs, (smax_ind - smin_ind + 1, ind_max-ind_min+1))

for si in range(smin_ind, smax_ind+1):
    for j, ci in enumerate(cind_arr):
        axs[si-1, j].plot(fac, np.exp(misfit_arr_all[si-1,j]), 'k', label='model')
        axs[si-1, j].set_title('$c_{%i}^{%i}$'%(ci, 2*si+1))

plt.tight_layout()

plt.savefig('model_vs_data_minfunc.png')
'''

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
        true_params[sind-1, ci] = GVARS.ctrl_arr_dpt_clipped[sind, cind]

#-------------saving the sigma in model params-------------------#
carr_sigma = np.zeros_like(true_params)

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        carr_sigma[sind-1, ci] = GVARS.ctrl_arr_sig_clipped[sind, cind]

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

# this is essentially the model function
pred += true_params_flat @ noc_diag_flat

# testing if the model still works after absorbing some ctrl points
# into the fixed part.
np.testing.assert_array_almost_equal(pred, eigvals_model)

#-------------saving miscellaneous files-------------------#
np.save('fixed_part.npy', diag_evals_fixed)
np.save('param_coeff_flat.npy', noc_diag_flat)
np.save('true_params_flat.npy', true_params_flat)
np.save('model_params_sigma.npy', carr_sigma_flat)
np.save('data_model.npy', eigvals_model)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)

sys.exit()

#-----------------generating the 2D pdfs-------------------#

true_params_flat = true_params.flatten(order='F')
num_params = len(true_params_flat)
noc_diag = np.asarray(noc_diag)
noc_diag = np.reshape(noc_diag, (num_params, -1), order='F')
noc_diag = jnp.asarray(noc_diag)
true_params_flat = jnp.asarray(true_params_flat)

def get_posterior_grid2d(pc1, pc2):
    fac = jnp.linspace(0.1, 1.9, N)
    misfit_arr = jnp.zeros((N, N))
    
    fac_nonpc = jnp.ones(num_params)
    fac_nonpc = jidx_update(fac_nonpc, jidx[pc1], 0.0)
    fac_nonpc = jidx_update(fac_nonpc, jidx[pc2], 0.0)
    fac = jnp.linspace(0.01, 1.9, N)
    
    fac_params_nonpc = fac_nonpc * true_params_flat

    # {{{ def true_func_i(i, misfits):
    def true_func_i(i, misfits):
        def true_func_j(j, misfits):
            pred = diag_evals_fixed +\
                   noc_diag[pc1] * true_params_flat[pc1] * fac[i] +\
                   noc_diag[pc2] * true_params_flat[pc2] * fac[j] +\
                   fac_params_nonpc @ noc_diag
            pred = jax.lax.cond(pc1==pc2,
                                lambda __: pred - noc_diag[pc2] *\
                                true_params_flat[pc2] * fac[j],
                                lambda __: pred, 
                                operand=None)
            misfits = jidx_update(misfits, jidx[i, j], 
                    -0.5*np.sum((eigvals_model - pred)**2)/len_data)
            return misfits

        return foril(0, N, true_func_j, misfits)
    # }}} true_func_i(i, misfits)

    misfits = foril(0, N, true_func_i, misfit_arr)
    return fac, misfits

# jitting the function
_get_posterior_grid2d = jax.jit(get_posterior_grid2d)

# plotting
fig, axs = plt.subplots(nrows=num_params, 
                        ncols=num_params, 
                        figsize=(10, 10))
for i in range(num_params):
    for j in range(i+1, num_params):
        thaxs = axs[j, i]
        fac, misfit_2d = _get_posterior_grid2d(i, j)
        facmin = fac.min()
        facmax = fac.max()
        plotval = np.exp(misfit_2d)

        im = thaxs.imshow(plotval, extent=[facmin*true_params_flat[i],
                                           facmax*true_params_flat[i],
                                           facmin*true_params_flat[j],
                                           facmax*true_params_flat[j]],
                          aspect=abs(true_params_flat[i]/true_params_flat[j]))
        thaxs.plot(true_params_flat[i], true_params_flat[j], 'xr')
        plt.colorbar(im, ax=thaxs)
        thaxs.set_title(f"c{i}-c{j}")

fig.tight_layout()
plt.savefig('2D_pdf.png')
