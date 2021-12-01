from collections import namedtuple
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

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
# defining the namedtuple containing tuples
# this will be used as a static argument
GVARS_ = namedtuple('GVARS',
                   ['a','b'])
a = ((1, 2), (3, 4))
b = (1.0, 2.0, 3.0, 4.0)
GVARS = GVARS_(a, b)

#========================================================================
@partial(jax.jit, static_argnums=(0,1))
def func1_w_staticargs(a_static, b_static):
    a_new = np.asarray(a_static) + 1
    b_new = np.asarray(b_static) + 1
    
    return a_new, b_new

@partial(jax.jit, static_argnums=(0,1))
def func2_w_staticargs(a_static, b_static):
    a_new = np.asarray(a_static) - 1
    b_new = np.asarray(b_static) - 1

    return a_new, b_new

def model():
    for i in range(1):
        # calling the function with two static arguments
        print('Starting loop')
        a_static, b_static = func2_w_staticargs(GVARS.a, GVARS.b)

        print('Printing as arrays: \n', a_static, b_static)
        
        # converting to arrays to tuple to pass as static arguments
        # to the next function call 
        a_static_tuple = tuple(map(tuple, a_static))
        b_static_tuple = tuple(b_static)
        GVARS_new_ = namedtuple('GVARS_new',
                               ['a','b'])
        GVARS_new = GVARS_new_(a_static_tuple, b_static_tuple)
        
        print('Printing as tuples: \n', GVARS_new.a, GVARS_new.b)
        print('Hash values :,\n', GVARS_new.a.__hash__(), GVARS_new.b.__hash__())

        # second function called with tuples passed as static arguments
        a_static, b_static = func2_w_staticargs(GVARS_new.a, GVARS_new.b)

        fac = 1.0
        eig_sample = np.asarray(GVARS_new.b)

    # eig_sample = numpyro.deterministic('eig', eig_mcmc_func(w1=w1, w3=w3, w5=w5))
    return numpyro.sample('obs', dist.Normal(eig_sample, np.ones_like(b)), obs=np.asarray(b))


# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(12)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=50, num_samples=100)
mcmc.run(rng_key_)
