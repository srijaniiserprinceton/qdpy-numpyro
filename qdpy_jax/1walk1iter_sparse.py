import numpy as np
import time
import sys
from collections import namedtuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.lax import fori_loop as foril
jidx = jax.ops.index
jidx_update = jax.ops.index_update

# new package in jax.numpy
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

#jax.config.update("jax_log_compiles", 1)
jax.config.update('jax_platform_name', 'cpu')

# enabling 64 bits
from jax.config import config
config.update('jax_enable_x64', True)

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nmults = len(GVARS.n0_arr) # total number of central multiplets

noc_hypmat_all_sparse, fixed_hypmat_all_sparse,\
    ell0_nmults, omegaref_nmults = precompute.build_hypmat_all_cenmults()

'''
noc_hypmat = tuple(map(tuple, (map(tuple, noc_hypmat_all_sparse))))
fixed_hypmat = tuple(fixed_hypmat_all_sparse)
'''

# necessary arguments to pass to build full hypermatrix
len_s = GVARS.wsr.shape[0]

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
                                                  GVARS.ctrl_arr_dpt_clipped,
                                                  GVARS.nc, len_s)
        
        # finding the eigenvalues of hypermatrix
        hypmat_dense = hypmat.todense()
        eigvals, __ = jnp.linalg.eigh(hypmat_dense)
        eigvalsum = jnp.sum(eigvals)
        totalsum += eigvalsum #+ elementsum
    return hypmat_dense


def eigval_sort_slice(eigval, eigvec):
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = np.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)

    return eigval[eigbasis_sort]

def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals


# jitting model
model_ = jax.jit(model)

# compiling
t1c = time.time()
hypmat = model_().block_until_ready()
# hypmat_eigs = get_eigs(hypmat.todense())[:401]/2./omegaref_nmults[0]*GVARS.OM*1e6
t2c = time.time()

print(f'Time for compilation in seconds: {(t2c-t1c):.3f}')

t1e = time.time()
Niter = 10
for i in range(Niter):
    hypmat = model_().block_until_ready()
t2e = time.time()
print(f'Time for execution in seconds: {(t2e-t1e)/Niter:.3f}')



# plotting difference with qdpt.py
supmat_qdpt = np.load('../../qdPy/supmat_qdpt.npy')

sm1 = np.diag(hypmat)
sm2 = np.diag(supmat_qdpt.real)

plt.figure()

plt.plot(sm1 - sm2)

plt.savefig('supmat_diff.pdf')
