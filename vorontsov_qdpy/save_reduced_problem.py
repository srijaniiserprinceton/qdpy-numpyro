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
from vorontsov_qdpy import sparse_precompute as precompute
from vorontsov_qdpy import sparse_precompute_bkm as precompute_bkm
from vorontsov_qdpy import build_hypermatrix_sparse as build_hm_sparse
from vorontsov_qdpy import build_hypermatrix_bkm as build_hm_bkm

from jax.lib import xla_bridge
print('JAX using:', xla_bridge.get_backend().platform)

# the indices of ctrl points that we want to investigate
ind_min, ind_max = 0, 1
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

# length of data
nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)


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
noc_bkm_sparse, fixed_bkm_sparse, k_arr, p_arr = precompute_bkm.build_bkm_all_cenmults()
num_k = int((np.unique(k_arr)>0).sum())

# precomputing the supermatrix components
noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sparse_idx =\
                                precompute.build_hypmat_all_cenmults()
len_data = len(omega0_arr)

noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)
noc_bkm = np.asarray(noc_bkm_sparse)
fixed_bkm = np.asarray(fixed_bkm_sparse)

fixmat_shape = fixed_hypmat_all_sparse[0].shape
max_nbs = fixmat_shape[1]
len_mmax = fixmat_shape[2]
len_s = noc_hypmat_all_sparse.shape[1]
nc = noc_hypmat_all_sparse.shape[2]

fixed_hypmat = np.reshape(fixed_hypmat_all_sparse,
                          (nmults, max_nbs*max_nbs*len_mmax),
                          order='F')
fixed_bkm = np.reshape(fixed_bkm_sparse, (nmults, max_nbs*max_nbs*len_mmax), order='F')

noc_hypmat = np.reshape(noc_hypmat_all_sparse,
                        (nmults, len_s, nc, max_nbs*max_nbs*len_mmax),
                        order='F')
noc_bkm = np.reshape(noc_bkm_sparse,
                     (nmults, len_s, nc, max_nbs*max_nbs*len_mmax),
                     order='F')

sparse_idxs_flat = np.zeros((nmults, max_nbs*max_nbs*len_mmax, 2), dtype=int)
for i in range(nmults):
    sparse_idxs_flat[i] = np.reshape(sparse_idx[i],
                                     (max_nbs*max_nbs*len_mmax, 2),
                                     order='F')

fixed_hypmat_dense = sparse.coo_matrix((fixed_hypmat[0],
                                        (sparse_idxs_flat[0, ..., 0],
                                        sparse_idxs_flat[0, ..., 1]))).toarray()
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


for i in range(nmults):
    synth_supmat_sparse = build_hm_sparse.build_hypmat_w_c(noc_hypmat[i],
                                                           fixed_hypmat[i],
                                                           GVARS.ctrl_arr_dpt_clipped,
                                                           GVARS.nc, len_s)
    
    # densifying
    synth_supmat[i] = sparse.coo_matrix((synth_supmat_sparse,
                                         (sparse_idxs_flat[i, ..., 0],
                                          sparse_idxs_flat[i, ..., 1])),
                                        shape=(dim_hyper, dim_hyper)).toarray()

    # supmat in muHz
    synth_supmat[i] *= 1.0/2./omega0_arr[i]*GVARS.OM*1e6


# testing the difference with eigvals_model
# np.testing.assert_array_almost_equal(synth_eigvals, eigvals_model, decimal=12)

############ COMPARING AGAINST supmat_qdpt.npy ######################## 
len_hyper_arr = fixed_hypmat.shape[-1]

fixed_hypmat_sparse = np.zeros((nmults, max_nbs, max_nbs, len_mmax))
fixed_bkm_sparse = np.zeros((nmults, max_nbs, max_nbs, len_mmax))

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

    _fixmat_bkm = build_hm_bkm.build_hypmat_w_c(noc_bkm_sparse[i],
                                                fixed_bkm_sparse[i],
                                                c_fixed, nc, len_s)

    fixed_hypmat_sparse[i, ...] = _fixmat
    fixed_bkm_sparse[i, ...] = _fixmat_bkm

param_coeff = np.zeros((len_s,
                        len(cind_arr),
                        nmults,
                        max_nbs,
                        max_nbs,
                        len_mmax))

param_coeff_bkm = np.zeros((len_s,
                            len(cind_arr),
                            nmults,
                            max_nbs,
                            max_nbs,
                            len_mmax))


for i in range(nmults):
    for si in range(len_s):
        for ci, cind in enumerate(cind_arr):
            param_coeff[si, ci, i, ...] = noc_hypmat_all_sparse[i, si, cind, ...]
            param_coeff_bkm[si, ci, i, ...] = noc_bkm_sparse[i, si, cind, ...]

# saving the supermatrix components
np.savetxt('.dimhyper', np.array([dim_hyper]), fmt='%d')
np.save('fixed_part.npy', fixed_hypmat_sparse)
np.save('param_coeff.npy', param_coeff)
np.save('sparse_idx.npy', sparse_idx)
np.save('omega0_arr.npy', omega0_arr)
np.save('dom_dell_arr.npy', GVARS.dom_dell)
np.save('ell0_arr.npy', ell0_arr)
np.save('data_model.npy', eigvals_model)
np.save('cind_arr.npy', cind_arr)
np.save('sind_arr.npy', sind_arr)
np.save('true_params.npy', true_params)

# saving the V11 bkm components
np.save('noc_bkm.npy', param_coeff_bkm)
np.save('fixed_bkm.npy', fixed_bkm_sparse)
np.save('k_arr.npy', k_arr)
np.save('p_arr.npy', p_arr)
