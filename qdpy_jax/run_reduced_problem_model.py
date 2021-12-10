import os
num_chains = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import argparse
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
from qdpy_jax import jax_functions as jf
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
parser.add_argument("--chain_num", help="chain number",
                    type=int, default=1)
parser.add_argument("--warmup", help="number of warmup steps",
                    type=int, default=20)
PARSED_ARGS = parser.parse_args()


read_params = np.loadtxt(".n0-lmin-lmax.dat")
ARGS = {}
ARGS['n0'] = int(read_params[0])
ARGS['lmin'] = int(read_params[1])
ARGS['lmax'] = int(read_params[2])
ARGS['rth'] = read_params[3]
ARGS['knot_num'] = int(read_params[4])
ARGS['load_from_file'] = int(read_params[5])

GVARS = gvar_jax.GlobalVars(n0=ARGS['n0'],
                            lmin=ARGS['lmin'],
                            lmax=ARGS['lmax'],
                            rth=ARGS['rth'],
                            knot_num=ARGS['knot_num'],
                            load_from_file=ARGS['load_from_file'])

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)

# loading the files forthe problem
dim_hyper = int(np.loadtxt('.dimhyper'))

# true params from Antia wsr
true_params = np.load('true_params.npy')
eigvals_true = np.load('evals_model.npy')
acoeffs_true = np.load('acoeffs_true.npy')
acoeffs_sigma = np.load('acoeffs_sigma.npy')

ell0_arr = np.load('ell0_arr.npy')
omega0_arr = np.load('omega0_arr.npy')

cind_arr = np.load('cind_arr.npy')
smin_ind, smax_ind = np.load('sind_arr.npy')

param_coeff = np.load('param_coeff.npy')
sparse_idx = np.load('sparse_idx.npy')
fixed_part = np.load('fixed_part.npy')
param_coeff = param_coeff[smin_ind:smax_ind+1, ...]

# comparing the matrices with the true values
supmat_jax = np.sum((true_params[:, :, NAX, NAX] * param_coeff),
                    axis=(0,1)) + fixed_part

"""
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
"""

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
sparse_idx = jnp.asarray(sparse_idx)

num_params = len(cind_arr)


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
# param_coeff *= 1e-3

ell0_arr_jax = jnp.asarray(GVARS.ell0_arr)
omega0_arr_jax = jnp.asarray(omega0_arr)

# reshaping true_params and param_coeff

true_params = jnp.reshape(true_params, (nc * len_s,), 'F')
param_coeff = jnp.reshape(param_coeff, (nc * len_s, nmults, -1), 'F')

# moving axis to allow seamless jnp.dot
param_coeff = jnp.moveaxis(param_coeff, 0, 1)


# setting the prior limits
cmin = 0.8 * jnp.ones_like(true_params)# / 1e-3
cmax = 1.2 * jnp.ones_like(true_params)# / 1e-3

ctrl_limits = {}
ctrl_limits['cmin'] = cmin
ctrl_limits['cmax'] = cmax


def compare_model():
    # predicted a-coeficients                                                             
    eigvals_compute = jnp.array([])
    eigvals_acoeffs = jnp.array([])
    pred_acoeffs = jnp.zeros(num_j * nmults)

    pred = true_params @ param_coeff + fixed_part
    
    def loop_in_mults(mult_ind, pred_acoeff):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind], sparse_idx[mult_ind],
                                         shape=(dim_hyper, dim_hyper))
        _eigval_mult = get_eigs(pred_dense)/2./omegaref*GVARS.OM*1e6
        Pjl_local = Pjl[mult_ind]
        # pred_acoeff = (Pjl_local @ _eigval_mult[:2*ell0+1])/Pjl_norm[i]
        # eigvals_acoeffs = jnp.append(eigvals_acoeffs, pred_acoeff)                    
        # eigvals_compute = jnp.append(eigvals_compute, _eigval_mult)                     
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff


    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    # diff = eigvals_compute - eigvals_true                                              
    # print(f"Max(Pred - True): {abs(diff).max():.5e}")                                   
    return pred_acoeffs

compare_model_ = jax.jit(compare_model)

def model():
    c_arr = numpyro.sample(f'c_arr', dist.Uniform(cmin, cmax))

    pred_acoeffs = jnp.zeros(num_j * nmults)
    pred = (c_arr*true_params) @ param_coeff + fixed_part

    def loop_in_mults(mult_ind, pred_acoeff):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]
        pred_dense = sparse.bcoo_todense(pred[mult_ind], sparse_idx[mult_ind],
                                         shape=(dim_hyper, dim_hyper))
        _eigval_mult = get_eigs(pred_dense)/2./omegaref*GVARS.OM*1e6
        Pjl_local = Pjl[mult_ind]
        # pred_acoeff = (Pjl_local @ _eigval_mult[:2*ell0+1])/Pjl_norm[i]                 
        # eigvals_acoeffs = jnp.append(eigvals_acoeffs, pred_acoeff)                      
        # eigvals_compute = jnp.append(eigvals_compute, _eigval_mult)                     
        
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

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
    count = 0
    for i in range(len(true_params)):
        sample = samples['c_arr'][:, i] * true_params[i]
        obs = true_params[i] 
        print(f"[{obs:11.4e}] c_arr[{i}]: {sample.mean():.4e} +/- {sample.std():.4e}:" +
              f"error/sigma = {(sample.mean()-obs)/sample.std():8.3f}")
    return None


if __name__ == "__main__":
    # diff = compare_model_()
    # Start from this source of randomness. We will split keys for subsequent operations.    
    seed = int(123 + 100*np.random.rand())
    rng_key = random.PRNGKey(seed)
    rng_key, rng_key_ = random.split(rng_key)
    # sys.exit()

    # kernel = SA(model, adapt_state_size=200)
    kernel = NUTS(model, max_tree_depth=(5, 1),
                  find_heuristic_step_size=True)
    mcmc = MCMC(kernel,
                num_warmup=PARSED_ARGS.warmup,
                num_samples=PARSED_ARGS.maxiter,
                num_chains=1)  
    mcmc.run(rng_key_, extra_fields=('potential_energy',))
    pe = mcmc.get_extra_fields()['potential_energy']

    # extracting necessary fields for plotting
    mcmc_sample = mcmc.get_samples()
    keys = mcmc_sample.keys()

    metadata = {}
    metadata['n0'] = ARGS['n0']
    metadata['lmin'] = ARGS['lmin']
    metadata['lmax'] = ARGS['lmax']
    metadata['rth'] = GVARS.rth
    metadata['knot_num'] = GVARS.knot_num
    metadata['maxiter'] = PARSED_ARGS.maxiter

    output_data = {}
    output_data['samples'] = mcmc.get_samples()
    output_data['metadata'] = metadata
    output_data['ctrl_limits'] = ctrl_limits

    fname = f"output-{PARSED_ARGS.maxiter}-{PARSED_ARGS.chain_num:03d}"
    jf.save_obj(output_data, f"{GVARS.scratch_dir}/{fname}")
    print_summary(mcmc_sample, true_params)

    plot_samples = {}

    # putting the true params
    refs = {}
    # initializing the keys
    for idx in range(num_params):
        sind = idx % 2
        ci = int(idx//2)
        s = 2*sind + 3
        argstr = f"c{s}_{ci}"
        refs[argstr] = true_params[idx]
        plot_samples[argstr] = output_data['samples']['c_arr'][:, idx]

    ax = az.plot_pair(
        plot_samples,
        var_names=[key for key in plot_samples.keys()],
        kde_kwargs={"fill_last": False},
        kind=["scatter", "kde"],
        marginals=True,
        point_estimate="median",
        figsize=(10, 8),
        reference_values=refs,
        reference_values_kwargs={'color':"red",
                                 "marker":"o",
                                 "markersize":6}
    )
    plt.tight_layout()
    plt.savefig(f'{GVARS.scratch_dir}/corner-reduced-{PARSED_ARGS.chain_num:03d}.png')

