from collections import namedtuple
import numpy as np
import py3nj
import time
import sys

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jax import random, vmap

# new package in jax.numpy
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import load_multiplets
from qdpy_jax import jax_functions as jf
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import prune_multiplets

# importing pyro related packages
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro import handlers


from jax.config import config
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


#========================================================================

W1T = 1.
W3T = 1.
W5T = 1.


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

    sigma = numpyro.sample('sigma', dist.Uniform(1e-3, 0.1))
    eig_sample = jnp.array([])

    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)
        CENMULT_AND_NBS = jf.tree_map_CNM_AND_NBS(CENMULT_AND_NBS)

        SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
        SUBMAT_DICT = jf.tree_map_SUBMAT_DICT(SUBMAT_DICT)

        supmatrix = build_supermatrix(CENMULT_AND_NBS,
                                      SUBMAT_DICT,
                                      GVARS_PRUNED_ST,
                                      GVARS_PRUNED_TR,
                                      ctrl_arr)
        eig_sample = jnp.append(eig_sample,
                                get_eigs(supmatrix)[:2*ell0+1]/2/CENMULT_AND_NBS.omega_nbs[0])

    # eig_sample = numpyro.deterministic('eig', eig_mcmc_func(w1=w1, w3=w3, w5=w5))
    return numpyro.sample('obs', dist.Normal(eig_sample, sigma), obs=eigvals_true)


def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = build_supmat.eigval_sort_slice(eigvals, eigvecs)
    return eigvals
#========================================================================

eigvals_true = np.array([])
GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours = build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours

build_SUBMAT_INDICES = build_supmat.build_SUBMAT_INDICES

# initialzing the class instance for supermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()
build_supermatrix = build_supmat_funcs.get_func2build_supermatrix()

# extracting the pruned parameters for multiplets of interest
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                prune_multiplets.get_pruned_attributes(GVARS, GVARS_ST)

lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                     nl_idx_pruned,
                                     omega_pruned)

GVARS_PRUNED_TR = jf.create_namedtuple('GVARS_TR',
                                       ['r',
                                        'r_spline',
                                        'rth',
                                        'wsr',
                                        'U_arr',
                                        'V_arr',
                                        'wig_list',
                                        'ctrl_arr_up',
                                        'ctrl_arr_lo',
                                        'knot_arr',
                                        'eigvals_true'],
                                       (GVARS_TR.r,
                                        GVARS_TR.r_spline,
                                        GVARS_TR.rth,
                                        GVARS_TR.wsr,
                                        lm.U_arr,
                                        lm.V_arr,
                                        wig_list,
                                        GVARS_TR.ctrl_arr_up,
                                        GVARS_TR.ctrl_arr_lo,
                                        GVARS_TR.knot_arr,
                                        GVARS_TR.eigvals_true))

GVARS_PRUNED_ST = jf.create_namedtuple('GVARS_ST',
                                       ['s_arr',
                                        'nl_all',
                                        'nl_idx_pruned',
                                        'omega_list',
                                        'fwindow',
                                        'OM',
                                        'wig_idx',
                                        'rth_ind',
                                        'spl_deg'],
                                       (GVARS_ST.s_arr,
                                        lm.nl_pruned,
                                        lm.nl_idx_pruned,
                                        lm.omega_pruned,
                                        GVARS_ST.fwindow,
                                        GVARS_ST.OM,
                                        wig_idx,
                                        GVARS_ST.rth_ind,
                                        GVARS_ST.spl_deg))

nmults = len(GVARS.n0_arr)
eigvals_true = jnp.asarray(GVARS_TR.eigvals_true)

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(12)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=50, num_samples=100)
mcmc.run(rng_key_)

save_obj(mcmc.get_samples(), f"{GVARS_PATHS.scratch_dir}/samples")
