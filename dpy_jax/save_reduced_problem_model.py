from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)

import argparse
import jax
from jax import random
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril
import matplotlib.pyplot as plt
import sys

config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'gpu')
config.update('jax_enable_x64', True)

parser = argparse.ArgumentParser()

parser.add_argument("--n0", help="min angular degree",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
parser.add_argument("--rth", help="threshold radius",
                    type=float, default=0.98)
parser.add_argument("--knot_num", help="number of knots beyong rth",
                    type=int, default=10)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.rth}" + "\n" +
            f"{ARGS.knot_num}" + "\n" +
            f"{ARGS.load_mults}")

# new package in jax.numpy
from dpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions as jf
from dpy_jax import sparse_precompute as precompute
from dpy_jax import build_hypermatrix_sparse as build_hm_sparse

from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)


######### parameters needed to be changed ###############

# the indices of ctrl points that we want to investigate
ind_min, ind_max = 0, 3
cind_arr = np.arange(ind_min, ind_max + 1)

smin, smax = 3, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.array([smin_ind, smax_ind])
 
#########################################################

GVARS = gvar_jax.GlobalVars(lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            n0=ARGS.n0,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
eigvals_model = np.load("evals_model.npy")
eigvals_model = jnp.asarray(eigvals_model)
# eigvals_sigma = jnp.ones_like(GVARS_TR.eigvals_sigma)
eigvals_sigma = jnp.asarray(GVARS_TR.eigvals_sigma)
num_eigvals = len(eigvals_model)


noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

# length of data
len_data = len(omega0_arr)

nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

cmax = jnp.asarray(GVARS.ctrl_arr_up)
cmin = jnp.asarray(GVARS.ctrl_arr_lo)

cmax = jnp.array(1.1 * GVARS.ctrl_arr_dpt_clipped)
cmin = jnp.array(0.9 * GVARS.ctrl_arr_dpt_clipped)

ctrl_arr_dpt = jnp.asarray(GVARS.ctrl_arr_dpt_clipped)

ctrl_limits = {}
ctrl_limits['cmin'] = {}
ctrl_limits['cmax'] = {}

for i in range(cmax.shape[1]-4):
    ctrl_limits['cmin'][f'c1_{i}'] = cmin[0, i]
    ctrl_limits['cmin'][f'c3_{i}'] = cmin[1, i]
    ctrl_limits['cmin'][f'c5_{i}'] = cmin[2, i]
    ctrl_limits['cmax'][f'c1_{i}'] = cmax[0, i]
    ctrl_limits['cmax'][f'c3_{i}'] = cmax[1, i]
    ctrl_limits['cmax'][f'c5_{i}'] = cmax[2, i]
    

# checks if the forward problem is working fine
def get_delta_omega():
    cdpt = GVARS.ctrl_arr_dpt_clipped

    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                fixed_hypmat_all_sparse,
                                                cdpt, nc, len_s)

    delta_omega = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6
    delta_omega -= eigvals_model
    
    return delta_omega

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

        delta_omega_model /= eigvals_sigma
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

print(f"delta_omega = {get_delta_omega()}")

# 1D plot of the misfit as we scan across the terrain

fig, axs = plt.subplots(smax_ind - smin_ind + 1, ind_max-ind_min+1, figsize=(12, 8), sharex = True)
axs = np.reshape(axs, (smax_ind - smin_ind + 1, ind_max-ind_min+1))

for si in range(smin_ind, smax_ind+1):
    for j, ci in enumerate(cind_arr):
        axs[si-1, j].plot(fac, np.exp(misfit_arr_all[si-1,j]), 'k', label='model')
        axs[si-1, j].set_title('$c_{%i}^{%i}$'%(ci, 2*si+1))

plt.tight_layout()

plt.savefig('model_vs_data_minfunc.png')

#--------- SECTION TO STORE MATRICES FOR COLLAB PROBLEM ------------#

c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in range(smin_ind, smax_ind+1):
    for cind in cind_arr:
        c_fixed[sind, cind] = 0.0
        c_fixed[sind, cind] = 0.0

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

fac_sig = 1./2./omega0_arr*GVARS.OM*1e6/eigvals_sigma

# this is the fixed part of the diag
diag_evals_fixed = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                    fixed_hypmat_all_sparse,
                                                    c_fixed, nc, len_s).todense()
diag_evals_fixed *= fac_sig

# we just need to save the noc_diag corresponding to the two ctrl_pts set to zero
noc_diag = []

for sind in range(smin_ind, smax_ind+1):
    noc_diag_s = []
    for cind in cind_arr:
        noc_diag_s.append(noc_hypmat_all_sparse[sind][cind].todense() * fac_sig)
    noc_diag.append(noc_diag_s)

# scaling the data
eigvals_model *= 1./eigvals_sigma

# checking if the forward problem works with the above components
pred = diag_evals_fixed * 1.0

# adding the contribution from the fitting part
for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        pred += GVARS.ctrl_arr_dpt_clipped[sind, cind] *\
                noc_diag[sind-1][ci]

true_params = np.zeros((smax_ind - smin_ind + 1, ind_max - ind_min + 1))

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        true_params[sind-1, ci] = GVARS.ctrl_arr_dpt_clipped[sind, cind]

print('Pred - Data:\n', np.max(np.abs(pred - eigvals_model)))

np.save('fixed_part.npy', diag_evals_fixed)
np.save('param_coeff.npy', noc_diag)
np.save('data_model.npy', eigvals_model)
np.save('true_params.npy', true_params)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)

sys.exit()
################# generating the 2D pdfs #########################

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
