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

numpyro.set_platform(f'{sys.argv[1]}')
numpyro.set_host_device_count(f'{sys.argv[2]}')
# numpyro.set_host_device_count(80)

nchains = int(sys.argv[2])
# nchains = 10

# initializing parameters
N = 50; J = 2
X = random.normal(random.PRNGKey(seed = 123), (N, J))
weight = np.array([1.5, -2.8])
error = random.normal(random.PRNGKey(234), (N, )) # standard Normal
b = 10.5
y_obs = X @ weight + b + error
y = y_obs.reshape((N, 1))
X = jax.device_get(X) # convert jax array into numpy array
y = jax.device_get(y) # convert jax array into numpy array

# creating a matrix to solve the dummy eigenvalue problem
# constructing the sparse matrix                                                                 
mat_sparse = scipy.sparse.rand(2500,2500,density=0.5)
mat_dense = mat_sparse.todense()
# making symmetric matrix                                                                     
sym_mat_raw = mat_dense + mat_dense.T
sym_mat = jax.device_put(sym_mat_raw)


# setting up model
def model(X, y=None):
    # dummy eigenvalue problem
    # print('Solving eval')
    # w_np, __ = numpy.linalg.eigh(sym_mat_raw + numpy.mean(jax.device_get(X)))
    # w = jax.device_put(w_np)
    # w, __ = jax.scipy.linalg.eigh(sym_mat+np.mean(X))
    
    ndims = np.shape(X)[-1]
    ws = numpyro.sample('betas', dist.Normal(0.0,10*np.ones(ndims)))
    b = numpyro.sample('b', dist.Normal(0.0, 10.0))
    sigma = numpyro.sample('sigma', dist.Uniform(0.0, 10.0))
    mu = X @ ws + b # + 1e-30*np.mean(w)

    return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

# setting up the sampler
nuts_kernel = NUTS(model)
num_warmup, num_samples = 10, 500
mcmc = MCMC(nuts_kernel, num_warmup, num_samples, num_chains=nchains)

# sampling
mcmc.run(random.PRNGKey(123), X, y = y_obs)
#mcmc.get_samples()

betas = np.mean(mcmc.get_samples()['betas'], axis=0)
b = mcmc.get_samples()['b'].mean()

print(betas, b)
