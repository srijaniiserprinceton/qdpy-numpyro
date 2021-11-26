import os
num_chains = 3
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import sys

import jax
import jax.numpy as jnp
from jax import random
from jax.config import config
from jax.experimental import sparse
from jax.lax import fori_loop as foril
from jax.lax import dynamic_slice as jdc
from jax.lax import dynamic_update_slice as jdc_update
from jax.ops import index_update as jidx_update
from jax.ops import index as jidx
config.update('jax_enable_x64', True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA

print(jax.devices()) # printing JAX devices

from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse


ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)

# loading the files forthe problem
dim_hyper = int(np.loadtxt('.dimhyper'))

# true params from Antia wsr
true_params = np.load('true_params.npy')
eigvals_true = np.load('evals_model.npy')
omega0_arr = np.load('omega0_arr.npy')
acoeffs_sigma = np.load('acoeffs_sigma.npy')
acoeffs_true = np.load('acoeffs_true.npy')

cind_arr = np.load('cind_arr.npy')
smin_ind, smax_ind = np.load('sind_arr.npy')

def create_sparse_fixed(fixmat, fixmat_idx):
    fixed_mat_list = []
    for i in range(nmults):
        sidx = i*nmults*dim_hyper
        eidx = (i+1)*nmults*dim_hyper
        _fs = sparse.BCOO((fixmat[sidx:eidx],
                           fixmat_idx[sidx:eidx, :]),
                          shape=(dim_hyper, dim_hyper))
        fixed_mat_list.append(_fs)
    return fixed_mat_list


def create_sparse_noc(nocmat, nocmat_idx):
    noc_mat_list = []
    for i in range(nmults):
        sidx = i*nmults*dim_hyper
        eidx = (i+1)*nmults*dim_hyper
        nocmat_s = []
        for sind in range(smin_ind, smax_ind+1):
            nocmat_c = []
            for cind in cind_arr:
                _spmat = sparse.BCOO((nocmat[sind, cind, sidx:eidx],
                                      nocmat_idx[sind, cind, sidx:eidx, :]),
                                     shape=(dim_hyper, dim_hyper))
                nocmat_c.append(_spmat)
            nocmat_s.append(nocmat_c)
        noc_mat_list.append(nocmat_s)
    return noc_mat_list


param_coeff = np.load('param_coeff.npy')
param_coeff_idx = np.load('param_coeff_idx.npy')
param_coeff_sparse = create_sparse_noc(param_coeff, param_coeff_idx)

fixed_part = np.load('fixed_part.npy')
fixed_part_idx = np.load('fixed_part_idx.npy')
fixed_part_sparse = create_sparse_fixed(fixed_part, fixed_part_idx)

# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load('RL_poly.npy')
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
Pjl = RL_poly[:, smin:smax+1:2, :]

# calculating the normalization for Pjl apriori
# shape (nmults, num_j)
Pjl_norm = np.zeros((nmults, Pjl.shape[1]))
for mult_ind in range(nmults):
    Pjl_norm[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)


# number of s to fit
len_s = true_params.shape[0]
# number of c's to fit
nc = true_params.shape[1]

# converting to device array
Pjl = jnp.asarray(Pjl)
# data = jnp.asarray(data)
true_params = jnp.asarray(true_params)
param_coeff = jnp.asarray(param_coeff)
fixed_part = jnp.asarray(fixed_part)
acoeffs_sigma = jnp.asarray(acoeffs_sigma)
Pjl_norm = jnp.asarray(Pjl_norm)

num_params = len(cind_arr)

# setting the prior limits
cmin = 0.5 * true_params #/ 1e-3
cmax = 1.5 * true_params #/ 1e-3

# this is actually done in the function create_sparse_noc
# param_coeff *= 1e-3

def compare_model():
    # predicted a-coefficients
    eigvals_compute = jnp.array([])
    for i in range(nmults):
        pred = build_hm_sparse.build_hypmat_w_c(param_coeff_sparse[i],
                                                fixed_part_sparse[i],
                                                true_params, nc, len_s)
        ell0 = GVARS.ell0_arr[i]
        omegaref = omega0_arr[i]
        _eigval_mult = jnp.diag(pred.todense())[:2*ell0+1]/2./omegaref*GVARS.OM*1e6
        eigvals_compute = jnp.append(eigvals_compute, _eigval_mult)

    diff = eigvals_compute - eigvals_true
    print(f"Max(Pred - True): {abs(diff).max():.5e}")
    return diff


def model():
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)
    c3 = []
    c5 = []

    for i in range(num_params):
        c3.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[0,i], cmax[0,i])))
        c5.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[1,i], cmax[1,i])))

    c3 = jnp.asarray(c3)
    c5 = jnp.asarray(c5)
    c_arr = jnp.vstack((c3, c5))

    pred_acoeff = jnp.zeros(num_j*nmults)

    for i in range(nmults):
        pred = build_hm_sparse.build_hypmat_w_c(param_coeff_sparse[i],
                                                fixed_part_sparse[i],
                                                c_arr, nc, len_s)
        ell0 = GVARS.ell0_arr[i]
        omegaref = omega0_arr[i]
        _eigval_mult = get_eigs(pred.todense())[:2*ell0+1]/2./omegaref*GVARS.OM*1e6
        Pjl_local = Pjl[i][:, :2*ell0+1]
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl_local @ _eigval_mult)/Pjl_norm[i],
                                (i * num_j,))

    misfit_acoeffs = (pred_acoeffs - acoeffs_true)/acoeffs_sigma
    return numpyro.factor('obs', dist.Normal(0.0, 1.0).log_prob(misfit_acoeffs))


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


def print_summary(samples, ctrl_arr):
    keys = samples.keys()
    for key in keys:
        sample = samples[key]
        key_split = key.split("_")
        idx = int(key_split[-1])
        sidx = int((int(key_split[0][1])-1)//2)
        obs = ctrl_arr[sidx-1, idx] / 1e-3
        print(f"[{obs:11.4e}] {key}: {sample.mean():.4e} +/- {sample.std():.4e}:" +
              f"error/sigma = {(sample.mean()-obs)/sample.std():8.3f}")
    return None


if __name__ == "__main__":
    diff = compare_model()
    # Start from this source of randomness. We will split keys for subsequent operations.    
    seed = int(123 + 100*np.random.rand())
    rng_key = random.PRNGKey(seed)
    rng_key, rng_key_ = random.split(rng_key)

    #kernel = SA(model, adapt_state_size=200)
    kernel = NUTS(model, max_tree_depth=(20, 5))
    mcmc = MCMC(kernel,
                num_warmup=1500,
                num_samples=6000,
                num_chains=num_chains)  
    mcmc.run(rng_key_, extra_fields=('potential_energy',))
    pe = mcmc.get_extra_fields()['potential_energy']

    # extracting necessary fields for plotting
    mcmc_sample = mcmc.get_samples()
    keys = mcmc_sample.keys()

    # putting the true params
    refs = {}
    # initializing the keys
    for sind in range(smin_ind, smax_ind+1):
        s = 2*sind + 1
        for ci in range(num_params):
            refs[f"c{s}_{ci}"] = true_params[sind-1, ci] / 1e-3

    ax = az.plot_pair(
        mcmc_sample,
        var_names=[key for key in mcmc_sample.keys()],
        kde_kwargs={"fill_last": False},
        kind=["scatter", "kde"],
        marginals=True,
        point_estimate="median",
        figsize=(10, 8),
        reference_values=refs,
        reference_values_kwargs={'color':"red", "marker":"o", "markersize":6}
    )
    plt.tight_layout()
    plt.savefig('corner_reduced_prob.png')

    print_summary(mcmc_sample, true_params)
