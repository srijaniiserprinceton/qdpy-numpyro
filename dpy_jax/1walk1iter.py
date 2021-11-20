import argparse
from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
import sys

# enabling 64 bits and logging compilatino
config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}" + "\n" +
            f"{ARGS.load_mults}")

# importing local package 
import jax_functions as jf
import globalvars as gvar_jax
import sparse_precompute as precompute
import build_hypermatrix_sparse as build_hm_sparse

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            load_from_file=ARGS.load_mults)
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.wsr.shape[0]  # number of s

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                precompute.build_hypmat_all_cenmults()

# sys.exit()

def model():
    # building the entire hypermatrix
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  GVARS.ctrl_arr_dpt_clipped,
                                                  GVARS.nc, len_s)
    
    # finding the eigenvalues of hypermatrix
    diag_dense = diag_evals.todense()
    
    return diag_dense


def eigval_sort_slice(eigval, eigvec):
    """Sorts eigenvalues using the eigenvectors"""
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = jnp.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)
    return eigval[eigbasis_sort]


def get_eigs(mat):
    """Returns the sorted eigenvalues of a real symmetric matrix"""
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals

def compare_hypmat():
    diag = model_().block_until_ready()
    return diag
    import matplotlib.pyplot as plt
    import numpy as np
    # plotting difference with qdpt.py
    supmat_qdpt = np.load("supmat_qdpt.npy").real
    supmat_qdpt_201 = np.load("supmat_qdpt_201.npy").real

    sm1 = diag
    sm2 = np.append(np.diag(supmat_qdpt)[:401],
                    np.diag(supmat_qdpt_201)[:403])

    plt.figure(figsize=(10, 5))
    plt.plot(sm1 - sm2)
    print(f"Max diff = {abs(sm1 - sm2).max():.3e}")
    plt.savefig('supmat_diff.pdf')
    return diag

if __name__ == "__main__":
    model_ = jit(model)

    # compiling
    jf.time_run(model_, prefix="compilation")
    jf.time_run(model_, prefix="execution", Niter=100,
                block_until_ready=True)

    diag = compare_hypmat()
