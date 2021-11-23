import os
num_chains = 38
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"
import numpy as np
import jax
print(jax.devices())
from jax import random
from jax import jit
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA
from jax.config import config
config.update('jax_enable_x64', True)
jidx_update = jax.ops.index_update
import arviz as az
import sys

# loading the files forthe problem
data = np.load('data_model.npy')
true_params = np.load('true_params.npy')
param_coeff = np.load('param_coeff.npy')
fixed_part = np.load('fixed_part.npy')
cind_arr = np.load('cind_arr.npy')

# number of s to fit
len_s = true_params.shape[0]
# number of c's to fit
nc = true_params.shape[1]

# converting to device array
data = jnp.asarray(data)
true_params = jnp.asarray(true_params)
param_coeff = jnp.asarray(param_coeff)
fixed_part = jnp.asarray(fixed_part)

# checking that the loaded data are correct
pred = fixed_part * 1.0

# adding the contribution from the fitting part
for sind in range(1,3):
    for ci, cind in enumerate(cind_arr):
        pred += true_params[sind-1, ci] * param_coeff[sind-1][ci]

# these arrays should be very close
np.testing.assert_array_almost_equal(pred, data)

num_params = len(cind_arr)

# setting the prior limits
cmin = 0.1 * true_params / 1e-3
cmax = 1.9 * true_params / 1e-3
param_coeff *= 1e-3

def model():
    # sampling from a uniform prior
    c3 = []
    c5 = []

    for i in range(num_params):
        c3.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[0,i], cmax[0,i])))
        c5.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[1,i], cmax[1,i])))
        # c3.append(numpyro.sample(f'c3_{i}', dist.Uniform(0.90, 1.10)))
        # c5.append(numpyro.sample(f'c5_{i}', dist.Uniform(0.90, 1.10)))

    c3 = jnp.asarray(c3)
    c5 = jnp.asarray(c5)
    # c3 = c3*true_params[0, i]
    # c5 = c5*true_params[1, i]
    
    pred = fixed_part - data + c3 @ param_coeff[0] + c5 @ param_coeff[1]
    return numpyro.factor('obs', dist.Normal(0.0, 1.0).log_prob(pred))


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
kernel = NUTS(model,
              max_tree_depth=(20, 5))
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
for sind in range(1, 3):
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
