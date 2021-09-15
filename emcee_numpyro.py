# miscellaneous imports                                                    
import jax
from jax import random
from jax import grad, jit
import jax.numpy as np # using jax.numpy instead                           
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import scipy.sparse
import sys
import numpy


# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534
# f_true = 1.0

# initializing parameters                                                
N = 50; J = 2
X = random.normal(random.PRNGKey(seed = 123), (N, J))
weight = np.array([m_true, 10*m_true])
# weight = np.zeros((1,1)) + m_true
error = 0.1 * random.normal(random.PRNGKey(234), (N, )) # standard Normal
                                              
y_obs = f_true * (X @ weight + b_true) + error*0.01

# setting up model                                                         
def model(X, y=None):
    ndims = np.shape(X)[-1]
    ws = numpyro.sample('betas', dist.Normal(0.0,10.0*np.ones(ndims)))
    b = numpyro.sample('b', dist.Normal(0.0, 10.0))
    sigma = numpyro.sample('sigma', dist.Uniform(0.0, 10.0))
    f = numpyro.sample('f', dist.Normal(0.0, 2.5))
    mu = f * (X @ ws + b)
    return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

# setting up the sampler                                                   
nuts_kernel = NUTS(model)
num_warmup, num_samples = 500, 1500
mcmc = MCMC(nuts_kernel, num_warmup, num_samples, num_chains=1)

# sampling                                                                 
mcmc.run(random.PRNGKey(240), X, y = y_obs)

# printing the NUTS summary
print(mcmc.print_summary())
