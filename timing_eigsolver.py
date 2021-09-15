# miscellaneous imports                                                  
import jax
from jax import random
from jax import grad, jit
import jax.numpy as np # using jax.numpy instead                         
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import scipy.sparse
import sys

import timeit
import time

numpyro.set_platform(f'{sys.argv[1]}')

# creating a matrix to solve the dummy eigenvalue problem                 
# constructing the sparse matrix                                           
mat_sparse = scipy.sparse.rand(3000,3000,density=0.5)
mat_dense = mat_sparse.todense()
# making symmetric matrix                                               
sym_mat_raw = mat_dense + mat_dense.T
sym_mat = jax.device_put(sym_mat_raw)

def model(X):
    Y,Z = jax.scipy.linalg.eigh(sym_mat+X)
    Y.block_until_ready()
    Z.block_until_ready()
    # return np.mean(Y)

# jitting the function
jax_model = jax.jit(model)
# first run
t1 = time.time()
__ = jax_model(1)
t2 = time.time()
print('First run: ',(t2-t1))

# timing CPU
dummy_sum = 0.0
Niter = 10
t1 = time.time()
for __ in range(Niter): dummy_sum += jax_model(1)
t2 = time.time()
time_cpu = (t2 - t1)/Niter
print(f'Time in {sys.argv[1]}: {time_cpu}')
print('Dummy sum: ', dummy_sum)
