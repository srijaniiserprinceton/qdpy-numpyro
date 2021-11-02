from collections import namedtuple
import numpy as np
import py3nj
import time
import sys

import jax
import jax.tree_util as tu
from jax import random, vmap
import jax.numpy as jnp

# new package in jax.numpy
from qdpy_jax import gnool_jit as gjit
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

jax.config.update('jax_platform_name', 'cpu')
numpyro.set_platform('cpu')

# enabling 64 bits

from jax.config import config
config.update('jax_enable_x64', True)

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

        # building the namedtuple for the central multiplet and its neighbours
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST)
        CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
        SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
        SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

        supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT,
                                    GVARS_PRUNED_ST, GVARS_PRUNED_TR)
        fac = 1.0
        fac *= (1 + w1*1e-3)
        fac *= (1 + w3*1e-3)
        fac *= (1 + w5*1e-3)
        fac /= 2.0*CENMULT_AND_NBS.omega_nbs[0]
        eig_sample = jnp.append(eig_sample, get_eigs(supmatrix)*fac)

    # eig_sample = numpyro.deterministic('eig', eig_mcmc_func(w1=w1, w3=w3, w5=w5))
    return numpyro.sample('obs', dist.Normal(eig_sample, sigma), obs=eigvals_true)

def eig_mcmc_func(w1=None, w3=None, w5=None):
    # eigs_combined = np.random.rand(2005)
    eigs_combined = np.array([1.0])
    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]

        # building the namedtuple for the central multiplet and its neighbours
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST)
        CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
        SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
        SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

        # supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT,
        #                             GVARS_PRUNED_ST, GVARS_PRUNED_TR)
        # fac = 1.0
        # fac *= (1 + w1*1e-3)
        # fac *= (1 + w3*1e-3)
        # fac *= (1 + w5*1e-3)
        # fac /= 2.0*CENMULT_AND_NBS.omega_nbs[0]
        # eigs_combined = jnp.append(eigs_combined, get_eigs(supmatrix)*fac)
        # eigs_combined = eigs_combined*fac
    return eigs_combined


def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    return eigvals


# slices out the unique nl, nl_idx and omega from
# from the arguments nl, omega which may contain repetitions
def get_pruned_multiplets(nl, omega, nl_all):
    n1 = nl[:, 0]
    l1 = nl[:, 1]

    omega_pruned = [omega[0]]
    
    nl_idx_pruned = [nl_all.tolist().index([nl[0, 0], nl[0, 1]])]
    nl_pruned = nl[0, :].reshape(1, 2)

    for i in range(1, len(n1)):
        try:
            nl_pruned.tolist().index([n1[i], l1[i]])
        except ValueError:
            nl_pruned = np.concatenate((nl_pruned,
                                        nl[i, :].reshape(1, 2)), 0)
            omega_pruned.append(omega[i])
            nl_idx_pruned.append(nl_all.tolist().index([nl[i, 0], nl[i, 1]]))
    return nl_pruned, nl_idx_pruned, omega_pruned

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours_ = \
    gjit.gnool_jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                   static_array_argnums = (0, 1, 2))

build_SUBMAT_INDICES_ = gjit.gnool_jit(build_supmat.build_SUBMAT_INDICES,
                                       static_array_argnums=(0,))

# initialzing the class instance for supermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()    
build_supermatrix_ = gjit.gnool_jit(build_supmat_funcs.get_func2build_supermatrix(),
                                    static_array_argnums=(0, 1, 2))

# COMPILING JAX
# looping over the ells
t1c = time.time()

# extracting the pruned parameters for multiplets of interest
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx = \
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

for i in range(nmults):
    n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

    supmatrix = build_supermatrix_(CENMULT_AND_NBS,
                                   SUBMAT_DICT,
                                   GVARS_PRUNED_ST,
                                   GVARS_PRUNED_TR).block_until_ready()
    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
t2c = time.time()
print(f'Time taken in seconds for compilation of {nmults} multiplets' +
      f' =  {t2c-t1c:.2f} seconds')

# EXECUTING JAX
t1e = time.time()
print("--------------------------------------------------")
eigvals_true = np.array([])

for i in range(nmults):
    n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]

    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT,
                                   GVARS_PRUNED_ST, GVARS_PRUNED_TR).block_until_ready()
    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
    eigvals_true = jnp.append(eigvals_true, np.diag(supmatrix))
t2e = time.time()

factor4niter = 1500 * 200./3600.
t_projected_jit = (t2e-t1e) / nmults * factor4niter
t_projected_eigval = 2. * factor4niter
print(f'Time taken in seconds by jax-jitted execution' +
      f' of entire simulation (1500 iterations) = {t_projected_jit:.2f} hours')

print(f'Assuming 2 seconds per eigenvalue problem solving, the ' +
      f'total time taken for EV solver (1500 iterations) = {t_projected_eigval:.2f} hours')

print('------------------')
print(f'Total time taken (1500 iterations) = ' +
      f'{(t_projected_jit + t_projected_eigval)/24.:.2f} days')

print(f'Fraction of time taken for setting up EV = ' +
      f'{t_projected_jit/t_projected_eigval:.3f}')



# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(12)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=50, num_samples=100)#, num_chains=5)
