import argparse
from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update

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
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}")

# importing local package 
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as precompute
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax)
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.wsr.shape[0]  # number of s

noc_hypmat_all_sparse, fixed_hypmat_all_sparse,\
    ell0_nmults, omegaref_nmults = precompute.build_hypmat_all_cenmults()


def model():
    totalsum = 0.0
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
        totalsum += eigvalsum
    return hypmat_dense


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
    hypmat = model_().block_until_ready()
    import matplotlib.pyplot as plt
    import numpy as np
    # plotting difference with qdpt.py
    supmat_qdpt = np.load("supmat_qdpt.npy").real

    sm1 = np.diag(hypmat)
    sm2 = np.diag(supmat_qdpt)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs = axs.flatten()
    axs[0].plot(sm1 - sm2)
    im = axs[1].imshow(hypmat - supmat_qdpt)
    plt.colorbar(im, ax=axs[1])
    print(f"Max diff = {abs(hypmat - supmat_qdpt).max():.3e}")
    fig.savefig('supmat_diff.pdf')
    return hypmat

if __name__ == "__main__":
    model_ = jit(model)

    # compiling
    jf.time_run(model_, prefix="compilation")
    jf.time_run(model_, prefix="execution", Niter=10,
                block_until_ready=True)

    # hypmat = compare_hypmat()
