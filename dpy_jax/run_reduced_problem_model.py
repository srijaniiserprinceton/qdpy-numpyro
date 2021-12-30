import os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import arviz as az
import sys

import jax
from jax import random
from jax import jit
from jax.lax import fori_loop as foril
from jax.lax import dynamic_slice as jdc
from jax.lax import dynamic_update_slice as jdc_update
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
print(jax.devices())

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA

from qdpy_jax import globalvars as gvar_jax

#----------------setting the number of chains to be used-----------------#
num_chains = 1
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
#------------------------------------------------------------------------#
ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

#-------------loading precomputed files for the problem-------------------# 
data = np.load('data_model.npy')
true_params_flat = np.load('true_params_flat.npy')
param_coeff_flat = np.load('param_coeff_flat.npy')
fixed_part = np.load('fixed_part.npy')
acoeffs_sigma_HMI = np.load('acoeffs_sigma_HMI.npy')
acoeffs_HMI = np.load('acoeffs_HMI.npy')
cind_arr = np.load('cind_arr.npy')
sind_arr = np.load('sind_arr.npy')
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1) 
RL_poly = np.load('RL_poly.npy')

#------------------------------------------------------------------------# 

nmults = len(GVARS.ell0_arr)
num_j = len(GVARS.s_arr)
dim_hyper = 2 * np.max(GVARS.ell0_arr) + 1
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
# number of s to fit
len_s = len(sind_arr)
# number of c's to fit
nc = len(cind_arr)

# slicing the Pjl correctly in angular degree s
Pjl = RL_poly[:, smin:smax+1:2, :]

#------------------------------------------------------------------------#
# calculating the denominator of a-coefficient converion apriori
# shape (nmults, num_j)
aconv_denom = np.zeros((nmults, Pjl.shape[1]))
for mult_ind in range(nmults):
    aconv_denom[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

#-------------------------converting to device array---------------------# 
Pjl = jnp.asarray(Pjl)
data = jnp.asarray(data)
true_params_flat = jnp.asarray(true_params_flat)
param_coeff_flat = jnp.asarray(param_coeff_flat)
fixed_part = jnp.asarray(fixed_part)
acoeffs_HMI = jnp.asarray(acoeffs_HMI)
acoeffs_sigma_HMI = jnp.asarray(acoeffs_sigma_HMI)
aconv_denom = jnp.asarray(aconv_denom)

#----------------------making the data_acoeffs---------------------------# 
data_acoeffs = jnp.zeros(num_j*nmults)

def loop_in_mults(mult_ind, data_acoeff):
    data_omega = jdc(data, (mult_ind*dim_hyper,), (dim_hyper,))
    data_acoeff = jdc_update(data_acoeff,
                             (Pjl[mult_ind] @ data_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))
    
    return data_acoeff

data_acoeffs = foril(0, nmults, loop_in_mults, data_acoeffs)

#---------------checking that the loaded data are correct----------------#
pred = fixed_part * 1.0

# adding the contribution from the fitting part
pred += true_params_flat @ param_coeff_flat

# these arrays should be very close
np.testing.assert_array_almost_equal(pred, data)

#-------------checking that the acoeffs match correctly------------------#
pred_acoeffs = jnp.zeros(num_j * nmults)

def loop_in_mults(mult_ind, pred_acoeff):
    pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
    pred_acoeff = jdc_update(pred_acoeff,
                             (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                             (mult_ind * num_j,))

    return pred_acoeff

pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)

# these arrays should be very close
np.testing.assert_array_almost_equal(pred_acoeffs, data_acoeffs)

#----------------------------------------------------------------------#

# changing to the HMI acoeffs if doing this for real data 
# data_acoeffs = GVARS.acoeffs_true

# setting the prior limits
cmin = 0.5 * jnp.ones_like(true_params_flat)# / 1e-3
cmax = 1.5 * jnp.ones_like(true_params_flat)# / 1e-3
#param_coeff *= 1e-3

init_params = {}
init_params[f'c_arr'] = jnp.ones_like(true_params_flat)
ip_nt = namedtuple('ip', init_params.keys())(*init_params.values())

'''
param_coeff = jnp.reshape(param_coeff, (nc*len_s, -1), 'F')
true_params_flat = jnp.reshape(true_params, nc*len_s, 'F')
'''
sys.exit()

def model():
    # predicted a-coefficients
    pred_acoeffs = jnp.zeros(num_j * nmults)
    # sampling from a uniform prior
    # c1 = []
    c_arr = numpyro.sample(f'c_arr', dist.Uniform(cmin, cmax))
    c_arr = c_arr * true_params_flat

    pred = fixed_part + c_arr @ param_coeff_flat

    def loop_in_mults(mult_ind, pred_acoeff):
        pred_omega = jdc(pred, (mult_ind*dim_hyper,), (dim_hyper,))
        pred_acoeff = jdc_update(pred_acoeff,
                                (Pjl[mult_ind] @ pred_omega)/aconv_denom[mult_ind],
                                (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    misfit_acoeffs = (pred_acoeffs - data_acoeffs)/acoeffs_sigma_HMI

    return numpyro.factor('obs', dist.Normal(0.0, 1.0).log_prob(misfit_acoeffs))


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


# Start from this source of randomness. We will split keys for subsequent operations.    
seed = int(123 + 100*np.random.rand())
rng_key = random.PRNGKey(seed)
rng_key, rng_key_ = random.split(rng_key)

#kernel = SA(model, adapt_state_size=200)
init_strat = numpyro.infer.init_to_value(values=init_params)

kernel = NUTS(model,
              max_tree_depth=(20, 5))
              # adapt_step_size=False,
              # step_size=1e-3,
              # init_strategy=init_strat)
mcmc = MCMC(kernel,
            num_warmup=5000,
            num_samples=5000,
            num_chains=num_chains)  
mcmc.run(rng_key_,
         extra_fields=('potential_energy',))
pe = mcmc.get_extra_fields()['potential_energy']

# extracting necessary fields for plotting
mcmc_sample = mcmc.get_samples()
keys = mcmc_sample.keys()

plot_samples = {}

# putting the true params
refs = {}
# initializing the keys
for idx in range(len_s * nc):
    sind = idx % len_s 
    ci = int(idx//len_s)
    s = 2*sind + 3
    argstr = f"c{s}_{ci}"
    refs[argstr] = 1.0
    plot_samples[argstr] = mcmc_sample['c_arr'][:, idx]

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

plt.savefig('corner_reduced_prob.png')

# print_summary(mcmc_sample, true_params)
