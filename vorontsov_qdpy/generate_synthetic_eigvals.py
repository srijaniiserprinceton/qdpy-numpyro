import argparse
import numpy as np
from scipy import sparse

from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.lax import fori_loop as foril
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
import sys

# enabling 64 bits and logging compilatino
config.update("jax_log_compiles", 0)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--rth", help="threshold radius",
                    type=float, default=0.98)
parser.add_argument("--knot_num", help="number of knots beyond rth",
                    type=int, default=10)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.rth}" + "\n" +
            f"{ARGS.knot_num}" + "\n" +
            f"{ARGS.load_mults}")

# importing local package 
from qdpy_jax import jax_functions as jf
from qdpy_jax import globalvars as gvar_jax
from vorontsov_qdpy import sparse_precompute as precompute
from vorontsov_qdpy import build_hypermatrix_sparse as build_hm_sparse

GVARS = gvar_jax.GlobalVars(n0=ARGS.n0,
                            lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
__, GVARS_TR, __ = GVARS.get_all_GVAR()
nmults = len(GVARS.n0_arr)  # total number of central multiplets
len_s = GVARS.wsr.shape[0]  # number of s
np.save('acoeffs_sigma.npy', GVARS.acoeffs_sigma)
np.save('acoeffs_true.npy', GVARS.acoeffs_true)


noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_arr, omega0_arr, sparse_idx =\
                                precompute.build_hypmat_all_cenmults()
noc_hypmat_all_sparse = np.asarray(noc_hypmat_all_sparse)
fixed_hypmat_all_sparse = np.asarray(fixed_hypmat_all_sparse)

fixmat_shape = fixed_hypmat_all_sparse[0].shape
max_nbs = fixmat_shape[1]
len_mmax = fixmat_shape[2]
len_s = noc_hypmat_all_sparse.shape[1]
nc = noc_hypmat_all_sparse.shape[2]

fixed_hypmat = np.reshape(fixed_hypmat_all_sparse,
                          (nmults, max_nbs*max_nbs*len_mmax),
                          order='F')
noc_hypmat = np.reshape(noc_hypmat_all_sparse,
                        (nmults, len_s, nc, max_nbs*max_nbs*len_mmax),
                        order='F')

sparse_idxs_flat = np.zeros((nmults, max_nbs*max_nbs*len_mmax, 2), dtype=int)
for i in range(nmults):
    sparse_idxs_flat[i] = np.reshape(sparse_idx[i],
                                     (max_nbs*max_nbs*len_mmax, 2),
                                     order='F')

fixed_hypmat_dense = sparse.coo_matrix((fixed_hypmat[0],
                                        (sparse_idxs_flat[0, ..., 0],
                                        sparse_idxs_flat[0, ..., 1]))).toarray()
dim_hyper = fixed_hypmat_dense.shape[0]


# converting to numpy ndarrays from lists


def model():
    eigval_model = jnp.array([])
    acoeff_model = jnp.array([])
    
    for i in range(nmults):
        eigval_mult = np.zeros(dim_hyper)
        hypmat_sparse = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[i],
                                                         fixed_hypmat_all_sparse[i],
                                                         GVARS.ctrl_arr_dpt_clipped,
                                                         GVARS.nc, len_s)

        hypmat_flat = np.reshape(hypmat_sparse, max_nbs*max_nbs*len_mmax, order='F')

        # converting to dense
        hypmat = sparse.coo_matrix((hypmat_flat,
                                    (sparse_idxs_flat[i, ..., 0],
                                     sparse_idxs_flat[i, ..., 1])),
                                   shape=(dim_hyper, dim_hyper)).toarray()

        print(i, np.diag(hypmat))

        # solving the eigenvalue problem and mapping eigenvalues
        ell0 = ell0_arr[i]
        omegaref = omega0_arr[i]
        eigval_qdpt_mult = get_eigs(hypmat)[:2*ell0+1]/2./omegaref
        # eigval_qdpt_mult = np.diag(hypmat)[:2*ell0+1]/2./omegaref
        eigval_qdpt_mult *= GVARS.OM*1e6
        eigval_mult[:len(eigval_qdpt_mult)] = eigval_qdpt_mult

        Pjl_local = Pjl[i][:, :2*ell0+1]
        qdpt_acoeff = (Pjl_local @ eigval_qdpt_mult)/Pjl_norm[i]

        # storing the correct order of nmult
        eigval_model = jnp.append(eigval_model, eigval_mult)
        acoeff_model = jnp.append(acoeff_model, qdpt_acoeff)

    return eigval_model, acoeff_model

def eigval_sort_slice(eigval, eigvec):
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = jnp.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)
    return eigval[eigbasis_sort]


def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals

def get_eigvals_sigma(len_evals_true):
    '''Function to get the sigma from data
    for the frequency splittings.
    '''
    # storing the eigvals sigmas
    eigvals_sigma = jnp.array([])

    ellmax = np.max(GVARS.ell0_arr)
    start_ind_gvar = 0
    start_ind = 0

    for i, ell in enumerate(GVARS.ell0_arr):
        end_ind_gvar = start_ind_gvar + 2 * ell + 1
        # storing in the correct order of nmult
        eigvals_sigma = jnp.append(eigvals_sigma,
                                   GVARS_TR.eigvals_sigma[start_ind_gvar:
                                                          end_ind_gvar])
        start_ind_gvar += 2 * ell + 1

    return eigvals_sigma


def compare_hypmat():
    hypmat_sparse = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse[-1],
                                                     fixed_hypmat_all_sparse[-1],
                                                     GVARS.ctrl_arr_dpt_clipped,
                                                     GVARS.nc, len_s)
    hypmat_flat = np.reshape(hypmat_sparse, max_nbs*max_nbs*len_mmax, order='F')

    # converting to dense
    hypmat = sparse.coo_matrix((hypmat_flat,
                                (sparse_idxs_flat[-1, ..., 0],
                                 sparse_idxs_flat[-1, ..., 1])),
                                shape=(dim_hyper, dim_hyper)).toarray()

    supmat_qdpt = np.load(f"supmat_qdpt_200.npy")
    matsize = supmat_qdpt.shape[0]
    supmat_model = hypmat[:matsize, :matsize]
    diff = supmat_model - supmat_qdpt
    print(f"Max diff = {abs(diff).max()}")
    return supmat_qdpt, supmat_model



if __name__ == "__main__":
    # model_ = jit(model)
    sm_qdpt, sm_model = compare_hypmat()
    RL_poly = np.load('RL_poly.npy')
    smin = min(GVARS.s_arr)
    smax = max(GVARS.s_arr)
    Pjl = RL_poly[:, smin:smax+1:2, :]

    Pjl_norm = np.zeros((Pjl.shape[0],
                         Pjl.shape[1]))
    for mult_ind in range(Pjl.shape[0]):
        Pjl_norm[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

    eigvals_true, acoeffs_true = model()
    eigvals_sigma = get_eigvals_sigma(len(eigvals_true))
    print(f"num elements = {len(eigvals_true)}")
    np.save("evals_model.npy", eigvals_true) 
    np.save('eigvals_sigma.npy', eigvals_sigma)
    np.save('acoeffs_sigma.npy', GVARS.acoeffs_sigma)
    np.save('acoeffs_true.npy', acoeffs_true)
