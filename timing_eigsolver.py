# miscellaneous imports                                                  
import jax
from jax import random
from jax import grad, jit
import jax.numpy as jnp # using jax.numpy instead                         
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import scipy.sparse
import sys

import timeit
import time

dtype = 'complex64'

numpyro.set_platform(f'{sys.argv[1]}')

# creating a matrix to solve the dummy eigenvalue problem                 
# constructing the sparse matrix                                           
mat_sparse = scipy.sparse.rand(3000,3000,density=0.5,dtype=dtype)
mat_dense = mat_sparse.todense()
# making symmetric matrix                                               
sym_mat_raw = mat_dense + mat_dense.T
sym_mat = jax.device_put(sym_mat_raw)

eigh = jax.scipy.linalg.eigh
_eigh = jit(jax.scipy.linalg.eigh)

def model(X):
    Y,Z = eigh(sym_mat+X)
    return jnp.mean(Y)

def _model(X):
    Y,Z = _eigh(sym_mat+X)
    return jnp.mean(Y)


# jitting the function
# jax_model = jax.jit(model)
# first run
t1 = time.time()
__ = _model(1).block_until_ready()
t2 = time.time()
print('First run jitted _model(): ',(t2-t1))

t1 = time.time()
__ = model(1).block_until_ready()
t2 = time.time()
print('First run for model(): ',(t2-t1))


# timing CPU
dummy_sum = 0.0
Niter = 10
t1 = time.time()
for __ in range(Niter): dummy_sum += _model(1).block_until_ready()
t2 = time.time()
time_cpu = (t2 - t1)/Niter
print(f'Time in {sys.argv[1]}: {time_cpu}')
print('Dummy sum: ', dummy_sum)

