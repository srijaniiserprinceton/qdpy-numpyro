import numpy as np
import time
import sys
from collections import namedtuple

import jax
import jax.numpy as jnp

# new package in jax.numpy
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_supermatrix as build_supmat
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

jax.config.update("jax_log_compiles", 1)
jax.config.update('jax_platform_name', 'cpu')

# enabling 64 bits
from jax.config import config
config.update('jax_enable_x64', True)

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nmults = len(GVARS.n0_arr) # total number of central multiplets

noc_hypmat_all_sparse, fixed_hypmat_all_sparse,\
    ell0_nmults, omegaref_nmults = precompute.build_hypmat_all_cenmults()

# necessary arguments to pass to build full hypermatrix
len_s = GVARS.wsr.shape[0]
nc = len(GVARS.ctrl_arr_up)

def model():
    totalsum = 0.0
    '''
    #for i in range(nmults):
    def loop_over_mults(i, totalsum):
        # building the entire hypermatrix
        hypmat =\
            build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                  GVARS.ctrl_arr_dpt)
        
        # finding the eigenvalues of hypermatrix
        #eigvals, __ = jnp.linalg.eigh(hypmat)
        #eigvalsum = jnp.sum(eigvals)

        #totalsum += eigvalsum #+ elementsum
        totalsum += jnp.sum(hypmat)
        return totalsum
       
    totalsum = jax.lax.fori_loop(0, nmults, loop_over_mults, totalsum)
    '''
    for i in range(nmults):
        # building the entire hypermatrix
        hypmat = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                  fixed_hypmat_all_sparse[i],
                                                  GVARS.ctrl_arr_up,
                                                  nc, len_s)
        
        # finding the eigenvalues of hypermatrix
        eigvals, __ = jnp.linalg.eigh(hypmat.todense())
        eigvalsum = jnp.sum(eigvals)
        totalsum += eigvalsum #+ elementsum
    return hypmat

def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = build_supmat.eigval_sort_slice(eigvals, eigvecs)
    return eigvals


# jitting model
model_ = jax.jit(model)

# compiling
t1c = time.time()
hypmat = model_()#.block_until_ready()
t2c = time.time()

print('Time for compilation in seconds:', (t2c-t1c))


Niter = 10
t1e = time.time()
for i in range(Niter): 
    print(i)
    hypmat = model_().todense()
    hypmat_eigs = get_eigs(hypmat)[:401]/2./omegaref_nmults[0]*GVARS.OM*1e6
t2e = time.time()

m_arr = np.arange(-200, 201)
print(hypmat_eigs/m_arr)


print('Time for execution in seconds:', (t2e-t1e)/Niter)
