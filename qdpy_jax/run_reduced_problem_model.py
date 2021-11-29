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

NAX = jnp.newaxis

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
ell0_arr = np.load('ell0_arr.npy')
acoeffs_sigma = np.load('acoeffs_sigma.npy')
acoeffs_true = np.load('acoeffs_true.npy')

cind_arr = np.load('cind_arr.npy')
smin_ind, smax_ind = np.load('sind_arr.npy')

param_coeff = np.load('param_coeff.npy')
sparse_idx = np.load('sparse_idx.npy')
fixed_part = np.load('fixed_part.npy')
param_coeff = param_coeff[smin_ind:smax_ind+1, ...]


# comparing the matrices with the true values
supmat_jax = np.sum((true_params[:, :, NAX, NAX] * param_coeff),
                    axis=(0,1)) + fixed_part

for i in range(nmults):
    supmat_jax_dense = sparse.bcoo_todense(supmat_jax[i], sparse_idx[i],
                                        shape=(dim_hyper, dim_hyper))
    supmat_jax_dense *= 1./2./omega0_arr[i] * GVARS.OM * 1e6
    ell0 = ell0_arr[i]
    supmat_qdpt = np.load(f'supmat_qdpt_{ell0}.npy').real
    supmat_qdpt *= 1./2./omega0_arr[i]*GVARS.OM*1e6
    spsize = supmat_qdpt.shape[0]
    diff = supmat_qdpt - supmat_jax_dense[:spsize, :spsize]
    print(f"cenmult l0 = {ell0}: maxdiff = {abs(diff).max()}")
    # max_diff = np.diag(supmat_jax_dense)[:2005] - np.diag(supmat_qdpt_200)
plt.plot(diff)
plt.savefig('supmat_diff.pdf')
# sys.exit()

# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
# reshaping to (nmults x (smax+1) x dim_hyper)
RL_poly = np.load('RL_poly.npy')
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
Pjl_read = RL_poly[:, smin:smax+1:2, :]
Pjl = np.zeros((Pjl_read.shape[0],
                Pjl_read.shape[1],
                dim_hyper))
Pjl[:, :, :Pjl_read.shape[2]] = Pjl_read

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
cmin = 0.5 * true_params / 1e-3
cmax = 1.5 * true_params / 1e-3


# making the data_acoeffs
data_acoeffs = jnp.zeros(num_j*nmults)
ell0_arr = jnp.array(GVARS.ell0_arr)

def loop_in_mults(mult_ind, data_acoeff):
    ell0 = ell0_arr[mult_ind]
    data_omega = jdc(eigvals_true, (mult_ind*dim_hyper,), (dim_hyper,))
    Pjl_local = Pjl[mult_ind]
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl_local @ data_omega)/Pjl_norm[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

# this is actually done in the function create_sparse_noc
param_coeff *= 1e-3

def compare_model():
    # predicted a-coefficients
    eigvals_compute = jnp.array([])
    eigvals_acoeffs = jnp.array([])

    pred = (true_params[..., NAX, NAX]/1e-3 * param_coeff).sum(axis=(0, 1)) + fixed_part
    for i in range(nmults):
        _eigval_mult = np.zeros(dim_hyper)
        ell0 = GVARS.ell0_arr[i]
        omegaref = omega0_arr[i]
        pred_dense = sparse.bcoo_todense(pred[i], sparse_idx[i],
                                         shape=(dim_hyper, dim_hyper))
        _eigval_mult[:2*ell0+1] = np.diag(pred_dense)[:2*ell0+1]/2./omegaref*GVARS.OM*1e6
        eigvals_compute = jnp.append(eigvals_compute, _eigval_mult)
        Pjl_local = Pjl[i]
        pred_acoeff = (Pjl_local @ _eigval_mult)/Pjl_norm[i]
        eigvals_acoeffs = jnp.append(eigvals_acoeffs, pred_acoeff)

    diff = eigvals_compute - eigvals_true
    print(f"Max(Pred - True): {abs(diff).max():.5e}")
    return eigvals_acoeffs


def model():
    # predicted a-coefficients
    c3 = []
    c5 = []

    for i in range(num_params):
        c3.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[0,i], cmax[0,i])))
        c5.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[1,i], cmax[1,i])))

    c3 = jnp.asarray(c3)
    c5 = jnp.asarray(c5)
    c_arr = jnp.vstack((c3, c5))


    pred_acoeffs = jnp.zeros(num_j * nmults)
    pred = (c_arr[..., NAX, NAX] * param_coeff).sum(axis=(0, 1)) + fixed_part

    for i in range(nmults):
        ell0 = GVARS.ell0_arr[i]
        omegaref = omega0_arr[i]
        pred_dense = sparse.bcoo_todense(pred[i], sparse_idx[i],
                                          shape=(dim_hyper, dim_hyper))
        # _eigval_mult = get_eigs(pred.todense())[:2*ell0+1]/2./omegaref*GVARS.OM*1e6
        _eigval_mult = jnp.diag(pred_dense)/2./omegaref*GVARS.OM*1e6
        Pjl_local = Pjl[i][:, :2*ell0+1]
        pred_acoeffs = jdc_update(pred_acoeffs,
                                (Pjl_local @ _eigval_mult[:2*ell0+1])/Pjl_norm[i],
                                (i * num_j,))

    misfit_acoeffs = (pred_acoeffs - data_acoeffs)/acoeffs_sigma
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

    # kernel = SA(model, adapt_state_size=200)
    kernel = NUTS(model)#, max_tree_depth=(20, 5))
    mcmc = MCMC(kernel,
                num_warmup=100,
                num_samples=500,
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

