import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse
from jax.experimental import sparse as jsparse

import jax
from jax import random
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.lax import fori_loop as foril
from jax.ops import index_update as jidx_update


config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

# new package in jax.numpy
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from vorontsov_qdpy import sparse_precompute_bkm as precompute_bkm
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)

# the indices of ctrl points that we want to investigate
ind_min, ind_max = 0, 3
cind_arr = np.arange(ind_min, ind_max + 1)

smin, smax = 3, 5
smin_ind, smax_ind = (smin-1)//2, (smax-1)//2
sind_arr = np.array([smin_ind, smax_ind])

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

GVARS_PATHS, GVARS_TR, __ = GVARS.get_all_GVAR()

eigvals_model = np.load("evals_model.npy")
eigvals_model = jnp.asarray(eigvals_model)
eigvals_sigma = jnp.asarray(np.load('eigvals_sigma.npy'))
acoeffs_sigma = jnp.asarray(np.load('acoeffs_sigma.npy'))
num_eigvals = len(eigvals_model)

# generating the true parameters
true_params = np.zeros((smax_ind - smin_ind + 1,
                        ind_max - ind_min + 1))

for sind in range(smin_ind, smax_ind+1):
    for ci, cind in enumerate(cind_arr):
        true_params[sind-1, ci] = GVARS.ctrl_arr_dpt_clipped[sind, cind]

# precomputing the V11 bkm components
noc_bkm, fixed_bkm, k_arr, p_arr = precompute_bkm.build_bkm_all_cenmults()

# precomputing the supermatrix components
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sp_indices_all =\
    precompute.build_hypmat_all_cenmults()

# densifying fixed_hypmat to get dim_hyper
fixed_hypmat_dense = sparse.coo_matrix((fixed_hypmat_all_sparse[0], sp_indices_all[0])).toarray()

# length of data
nc = GVARS.nc
len_data = len(omega0_arr)
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)
dim_hyper = fixed_hypmat_dense.shape[0]

# functions to compute and correctly map the eigenvalues

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

############ COMPARING AGAINST THE eigvals_model ########################
# generating the synthetic data and comparing with eigval_model
nmults = len(GVARS.n0_arr)
len_s = GVARS.wsr.shape[0]

synth_supmat = np.zeros((nmults, dim_hyper, dim_hyper))
synth_eigvals = jnp.array([])

# converting from list to arrays
noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)


for i in range(nmults):
    synth_supmat_sparse = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                           fixed_hypmat_all_sparse[i],
                                                           GVARS.ctrl_arr_dpt_clipped,
                                                           GVARS.nc, len_s)
    
    # densifying
    synth_supmat[i] = sparse.coo_matrix((synth_supmat_sparse, sp_indices_all[i]),
                                        shape = (dim_hyper, dim_hyper)).toarray()
    # supmat in muHz
    synth_supmat[i] *= 1.0/2./omega0_arr[i]*GVARS.OM*1e6
    
    '''
    # extracting eigenvalues
    ell0 = ell0_arr[i]
    omegaref = omega0_arr[i]
    eigval_qdpt_mult = get_eigs(synth_supmat[i])[:2*ell0+1]/2./omegaref
    eigval_qdpt_mult *= GVARS.OM*1e6
    
    # storing in the correct order of nmult
    synth_eigvals = jnp.append(synth_eigvals, eigval_qdpt_mult)
    '''

# testing the difference with eigvals_model
# np.testing.assert_array_almost_equal(synth_eigvals, eigvals_model, decimal=12)

############ COMPARING AGAINST supmat_qdpt.npy ########################
"""

for il, ell in enumerate(ell0_arr):
    print(f'{il}: {ell}')
    supmat_qdpt = np.load(f'supmat_qdpt_{ell}.npy') / 2. / omega0_arr[il] * GVARS.OM * 1e6
    spsize = supmat_qdpt.shape[0]
    np.testing.assert_array_almost_equal(synth_supmat[il][:spsize, :spsize],
                                         supmat_qdpt, decimal=12, verbose=True)


"""
############ COMPARING AGAINST supmat_qdpt.npy ######################## 
len_hyper_arr = fixed_hypmat_all_sparse.shape[-1]

# converting to array and changing the axes to nmult X element_idx X xy-identifier
hypmat_idx = np.array(sp_indices_all, dtype=int)
hypmat_idx = np.moveaxis(hypmat_idx, 1, -1)

fixed_hypmat_sparse = np.zeros((nmults, len_hyper_arr))

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

c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()

# making the c_fixed coeffs for the variable params zero
for sind in range(smin_ind, smax_ind+1):
    for cind in cind_arr:
        c_fixed[sind, cind] = 0.0
        c_fixed[sind, cind] = 0.0

# this is the fixed part of the supermatrix
for i in range(nmults):
    _fixmat = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                               fixed_hypmat_all_sparse[i],
                                               c_fixed, nc, len_s)

    fixed_hypmat_sparse[i, :] = _fixmat

param_coeff = np.zeros((len_s,
                        len(cind_arr),
                        nmults,
                        len_hyper_arr))

for i in range(nmults):
    for si in range(len_s):
        for ci, cind in enumerate(cind_arr):
            param_coeff[si, ci, i, :] = noc_hypmat_all_sparse[i, si, cind, :]

# saving the supermatrix components
np.save('fixed_part.npy', fixed_hypmat_sparse)
np.save('param_coeff.npy', param_coeff)
np.save('sparse_idx.npy', hypmat_idx)
np.savetxt('.dimhyper', np.array([dim_hyper]), fmt='%d')
np.save('omega0_arr.npy', omega0_arr)
np.save('ell0_arr.npy', ell0_arr)
np.save('data_model.npy', eigvals_model)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)
np.save('true_params.npy', true_params)

# saving the V11 bkm components
np.save('noc_bkm.npy', noc_bkm)
np.save('fixed_bkm.npy', fixed_bkm)
np.save('k_arr.npy', k_arr)
np.save('p_arr.npy', p_arr)
sys.exit()
# sys.exit()

# we just need to save the noc_diag corresponding to the two ctrl_pts set to zero
noc_hypmat_sparse = np.zeros((len(GVARS.s_arr),
                              len(cind_arr),
                              nmults,
                              9*dim_hyper))
noc_hypmat_idx = np.zeros((len(GVARS.s_arr),
                           len(cind_arr),
                           nmults,
                           9*dim_hyper, 2),
                          dtype=int)

for i in range(nmults):
    for sind in range(smin_ind, smax_ind+1):
        for cind in cind_arr:
            _noc_sp = noc_hypmat_all_sparse[i][sind][cind]
            _noc_hypmat_sp = jsparse.BCOO.fromdense(_noc_sp.todense())
            _data = _noc_hypmat_sp.data
            _idx = _noc_hypmat_sp.indices
            _lendata = len(_data)
            noc_hypmat_sparse[sind, cind, i, :_lendata] = _data
            noc_hypmat_idx[sind, cind, i, :_lendata, :] = _idx

np.save('param_coeff.npy', noc_hypmat_sparse)
np.save('param_coeff_idx.npy', noc_hypmat_idx)
np.save('data_model.npy', eigvals_model)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)
np.save('true_params.npy', true_params)
sys.exit()


# checks if the forward problem is working fine
def get_delta_omega():
    eigval_all = np.array([])
    for i in range(nmults):
        diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                    fixed_hypmat_all_sparse[i],
                                                    GVARS.ctrl_arr_dpt_clipped, nc, len_s)
                                                    # true_params, nc, len_s)
        ell0 = GVARS.ell0_arr[i]
        eigval_compute = np.diag(diag_evals.todense())/2./omega0_arr[i]*GVARS.OM*1e6
        eigval_compute = eigval_compute[:2*ell0+1]
        eigval_all = np.append(eigval_all, eigval_compute)
    return eigval_all

pred = get_delta_omega()
# checking if the forward problem works with the above components
print(f"Pred - Data: {abs(pred - eigvals_model).max():.6e}")


############################################################################################
"""
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

print(f"delta_omega = {get_delta_omega()}")

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
############################################################################################


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

"""
