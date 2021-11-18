from collections import namedtuple
import numpy as np
import py3nj
import time
import sys

import os
num_chains = 1
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
#+"--xla_dump_to=/tmp/foo"

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jax import random, vmap
from jax.lax import fori_loop as foril
jidx = jax.ops.index
jidx_update = jax.ops.index_update

# new package in jax.numpy
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

# importing pyro related packages
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA
from numpyro import handlers


from jax.config import config
jax.config.update("jax_log_compiles", 1)
jax.config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)
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

nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

def model():
    # setting min and max value to be 0.1*true and 3.*true
    cmax = GVARS_TR.ctrl_arr_up
    cmin = GVARS_TR.ctrl_arr_lo

    c1_list = []
    c3_list = []
    c5_list = []
    for i in range(cmax.shape[1]):
        c1_list.append(numpyro.sample(f'c1_{i}', dist.Uniform(cmin[0, i], cmax[0, i])))
        c3_list.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[1, i], cmax[1, i])))
        c5_list.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[2, i], cmax[2, i])))

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

    eigbasis_sort = np.zeros(len(eigval), dtype=int)
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
kernel = SA(model)
mcmc = MCMC(kernel, num_warmup=50, num_samples=100, num_chains=num_chains)
mcmc.run(rng_key_)

save_obj(mcmc.get_samples(), f"{GVARS_PATHS.scratch_dir}/samples")
