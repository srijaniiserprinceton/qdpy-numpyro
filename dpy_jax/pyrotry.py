import argparse
from jax import random
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril
import sys

config.update("jax_log_compiles", 1)
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)

parser = argparse.ArgumentParser()

parser.add_argument("--n0", help="min angular degree",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--load_mults", help="load mults from file",
                    type=int, default=0)
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
parser.add_argument("--chain_num", help="chain number",
                    type=int, default=1)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.load_mults}")

# new package in jax.numpy
from dpy_jax import globalvars as gvar_jax
from dpy_jax import jax_functions as jf
from dpy_jax import sparse_precompute as precompute
from dpy_jax import build_hypermatrix_sparse as build_hm_sparse

# importing pyro related packages
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA
from numpyro.infer import init_to_sample, init_to_value
numpyro.set_platform('cpu')

GVARS = gvar_jax.GlobalVars(lmin=ARGS.lmin,
                            lmax=ARGS.lmax,
                            n0=ARGS.n0,
                            load_from_file=ARGS.load_mults)
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
eigvals_true = jnp.asarray(GVARS_TR.eigvals_true)
eigvals_sigma = jnp.asarray(GVARS_TR.eigvals_sigma)

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

'''
noc_hypmat = tuple(map(tuple, (map(tuple, noc_hypmat_all_sparse))))
fixed_hypmat = tuple(fixed_hypmat_all_sparse)
'''

nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

cmax = jnp.asarray(GVARS.ctrl_arr_up)
cmin = jnp.asarray(GVARS.ctrl_arr_lo)
ctrl_arr_dpt = jnp.asarray(GVARS.ctrl_arr_dpt_clipped)

def model():
    # setting min and max value to be 0.1*true and 3.*true
    c1_list = []
    c3_list = []
    c5_list = []

    for i in range(cmax.shape[1]-4):
        c1_list.append(numpyro.sample(f'c1_{i}', dist.Uniform(cmin[0, i], cmax[0, i])))
        c3_list.append(numpyro.sample(f'c3_{i}', dist.Uniform(cmin[1, i], cmax[1, i])))
        c5_list.append(numpyro.sample(f'c5_{i}', dist.Uniform(cmin[2, i], cmax[2, i])))

    for i in range(4):
        c1_list.append(0.0)
        c1_list.append(0.0)
        c1_list.append(0.0)

    ctrl_arr = [jnp.array(c1_list),
                jnp.array(c3_list),
                jnp.array(c5_list)]

    eig_sample = jnp.array([])

    
    # building the entire hypermatrix
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  ctrl_arr, nc, len_s)
    
    delta_omega = diag_evals.todense()/2./omega0_arr

    return numpyro.factor('obs', dist.Normal(delta_omega,
                                             eigvals_sigma).log_prob(eigvals_true))

'''
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
'''

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(ARGS.lmin+ARGS.lmax+ARGS.chain_num)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
#kernel = NUTS(model)
kernel = SA(model)#, init_strategy=init_to_value(values=ctrl_arr_dpt))
mcmc = MCMC(kernel, num_warmup=500, num_samples=ARGS.maxiter)
mcmc.run(rng_key_)

fname = f"samples-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
jf.save_obj(mcmc.get_samples(), f"{GVARS_PATHS.scratch_dir}/{fname}")
