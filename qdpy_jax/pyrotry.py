from jax import random
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril

config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

# new package in jax.numpy
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

# importing pyro related packages
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA
from numpyro.infer import init_to_sample, init_to_value
numpyro.set_platform('cpu')

import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
eigvals_true = jnp.asarray(GVARS_TR.eigvals_true)
eigvals_sigma = jnp.asarray(GVARS_TR.eigvals_sigma)

noc_hypmat_all_sparse, fixed_hypmat_all_sparse,\
    ell0_nmults, omegaref_nmults = precompute.build_hypmat_all_cenmults()

'''
noc_hypmat = tuple(map(tuple, (map(tuple, noc_hypmat_all_sparse))))
fixed_hypmat = tuple(fixed_hypmat_all_sparse)
'''

nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

cmax = jnp.asarray(GVARS.ctrl_arr_up)
cmin = jnp.asarray(GVARS.ctrl_arr_lo)
ctrl_arr_dpt = jnp.asarray(GVARS.ctrl_arr_dpt_clipped)

def model():
    # setting min and max value to be 0.1*true and 3.*true
    '''
    ctrl_arr = jnp.zeros((len_s, nc))

    def true_func(i, c_arr):
        c_arr = jidx_update(c_arr, jidx[0, i],
            numpyro.sample(f'c1_{i}', dist.Uniform(cmin[0, i], cmax[0, i])))
        c_arr = jidx_update(c_arr, jidx[1, i],
            numpyro.sample(f'c3_{i}', dist.Uniform(cmin[1, i], cmax[1, i])))
        c_arr = jidx_update(c_arr, jidx[2, i],
            numpyro.sample(f'c5_{i}', dist.Uniform(cmin[2, i], cmax[2, i])))
        return c_arr

    ctrl_arr = foril(0, nc-4, true_func, ctrl_arr)
    '''
    c1_list = []
    c3_list = []
    c5_list = []

    for i in range(cmax.shape[1]-4):
        c1_list.append(numpyro.sample(f'c1_{i}', dist.Uniform(cmin[0, i], cmax[0, i])))
        c3_list.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[1, i], cmax[1, i])))
        c5_list.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[2, i], cmax[2, i])))

    for i in range(4):
        c1_list.append(0.0)
        c1_list.append(0.0)
        c1_list.append(0.0)

    ctrl_arr = [jnp.array(c1_list),
                jnp.array(c3_list),
                jnp.array(c5_list)]

    eig_sample = jnp.array([])

    for i in range(nmults):
        # building the entire hypermatrix
        hypmat = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                  fixed_hypmat_all_sparse[i],
                                                  ctrl_arr, nc, len_s)

        ell0 = ell0_nmults[i]
        omegaref = omegaref_nmults[i]
        _eigval_mult = get_eigs(hypmat.todense())[:2*ell0+1]/2./omegaref
        eig_sample = jnp.append(eig_sample, _eigval_mult*GVARS.OM*1e6)

    return numpyro.factor('obs', dist.Normal(eig_sample,
                                             eigvals_sigma).log_prob(eigvals_true))


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


# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(12)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
#kernel = NUTS(model)
kernel = SA(model, init_strategy=init_to_value(values=ctrl_arr_dpt))
mcmc = MCMC(kernel, num_warmup=1500, num_samples=2000)
mcmc.run(rng_key_)

save_obj(mcmc.get_samples(), f"{GVARS_PATHS.scratch_dir}/samples")
