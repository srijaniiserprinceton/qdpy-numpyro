import numpy as np
import time
import sys

import jax
import jax.numpy as jnp

# new package in jax.numpy
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import precompute_and_load as precompute
from qdpy_jax import build_hypermatrix_nostatic as build_hm


jax.config.update("jax_log_compiles", 1)
jax.config.update('jax_platform_name', 'cpu')

# enabling 64 bits
from jax.config import config
config.update('jax_enable_x64', True)

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

nl_pruned_all, omega_pruned_all, HM_DICT =\
                            precompute.precompute(GVARS, GVARS_ST)

# total number of central multiplets
nmults = len(GVARS.n0_arr)

def model():
    totalsum = 0.0
    #for i in range(nmults):
    def loop_over_mults(i, totalsum):
        non_m_hypmat = build_hm.build_non_m_part(i, HM_DICT)
        # elementsum = jnp.sum(non_m_hypmat)
        
        eigvals, __ = jnp.linalg.eigh(non_m_hypmat)
        eigvalsum = jnp.sum(eigvals)

        totalsum += eigvalsum #+ elementsum

        return totalsum
       
    totalsum = jax.lax.fori_loop(0, nmults, loop_over_mults, totalsum)
    return totalsum
    
# jitting model
model_ = jax.jit(model)

# compiling
t1c = time.time()
model_().block_until_ready()
t2c = time.time()

print('Time for compilation in seconds:', (t2c-t1c))


Niter = 10
t1e = time.time()
for i in range(Niter): 
    print(i)
    model_().block_until_ready()
t2e = time.time()

print('Time for execution in seconds:', (t2e-t1e)/Niter)
