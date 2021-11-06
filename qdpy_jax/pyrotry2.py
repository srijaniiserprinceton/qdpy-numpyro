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


#========================================================================

W1T = 1.
W3T = 1.
W5T = 1.


def model():
    # setting min and max value to be 0.1*true and 3.*true
    w1min, w1max = .1*abs(W1T), 3.*abs(W1T)
    w3min, w3max = .1*abs(W3T), 3.*abs(W3T)
    w5min, w5max = .1*abs(W5T), 3.*abs(W5T)
    
    w1 = numpyro.sample('w1', dist.Uniform(w1min, w1max))
    w3 = numpyro.sample('w3', dist.Uniform(w3min, w3max))
    w5 = numpyro.sample('w5', dist.Uniform(w5min, w5max))
    
    sigma = numpyro.sample('sigma', dist.Uniform(0.1, 10.0))
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
                                       GVARS_PRUNED_TR)

        fac = 1.0
        fac *= (1 + w1*1e-3)
        fac *= (1 + w3*1e-3)
        fac *= (1 + w5*1e-3)
        fac /= 2.0*CENMULT_AND_NBS.omega_nbs[0]
        eig_sample = jnp.append(eig_sample, get_eigs(supmatrix)*fac)
        
    eigvals_true = jnp.ones_like(eig_sample) * eig_sample * 1.1
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
                                        'rth',
                                        'rmin_ind',
                                        'rmax_ind',
                                        'fac_up',
                                        'fac_lo',
                                        'wsr',
                                        'U_arr',
                                        'V_arr',
                                        'wig_list'],
                                       (GVARS_TR.r,
                                        GVARS_TR.rth,
                                        GVARS_TR.rmin_ind,
                                        GVARS_TR.rmax_ind,
                                        GVARS_TR.fac_up,
                                        GVARS_TR.fac_lo,
                                        GVARS_TR.wsr,
                                        lm.U_arr,
                                        lm.V_arr,
                                        wig_list))

GVARS_PRUNED_ST = jf.create_namedtuple('GVARS_ST',
                                       ['s_arr',
                                        'nl_all',
                                        'nl_idx_pruned',
                                        'omega_list',
                                        'fwindow',
                                        'OM',
                                        'wig_idx'],
                                       (GVARS_ST.s_arr,
                                        lm.nl_pruned,
                                        lm.nl_idx_pruned,
                                        lm.omega_pruned,
                                        GVARS_ST.fwindow,
                                        GVARS_ST.OM,
                                        wig_idx))

nmults = len(GVARS.n0_arr)

'''
# trying to jit model
model_ = jax.jit(model)
model()
model_()
print('model_() runs')
'''

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(12)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=50, num_samples=100)
mcmc.run(rng_key_)
