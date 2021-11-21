import argparse
import jax
from jax import random
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp
from jax.config import config
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update
from jax.lax import fori_loop as foril
import matplotlib.pyplot as plt
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
parser.add_argument("--rth", help="threshold radius",
                    type=float, default=0.98)
parser.add_argument("--knot_num", help="number of knots beyong rth",
                    type=int, default=10)
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
parser.add_argument("--chain_num", help="chain number",
                    type=int, default=1)
ARGS = parser.parse_args()

with open(".n0-lmin-lmax.dat", "w") as f:
    f.write(f"{ARGS.n0}" + "\n" +
            f"{ARGS.lmin}" + "\n" +
            f"{ARGS.lmax}"+ "\n" +
            f"{ARGS.rth}" + "\n" +
            f"{ARGS.knot_num}" + "\n" +
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
                            rth=ARGS.rth,
                            knot_num=ARGS.knot_num,
                            load_from_file=ARGS.load_mults)
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
eigvals_model = np.load("evals_model.npy")
eigvals_model = jnp.asarray(eigvals_model)
# eigvals_sigma = jnp.ones_like(GVARS_TR.eigvals_sigma)
eigvals_true = jnp.asarray(GVARS_TR.eigvals_true)
eigvals_sigma = jnp.asarray(GVARS_TR.eigvals_sigma)
num_eigvals = len(eigvals_true)

# the log prob shift due to the 1/sqrt(2pi sigma^2)
log_prob_shift = - jnp.log(jnp.sqrt(2*jnp.pi) * jnp.sum(eigvals_sigma))

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

# length of data
len_data = len(omega0_arr)

nc = GVARS.nc
len_s = len(GVARS.s_arr)
nmults = len(GVARS.n0_arr)

cmax = jnp.asarray(GVARS.ctrl_arr_up)
cmin = jnp.asarray(GVARS.ctrl_arr_lo)
ctrl_arr_dpt = jnp.asarray(GVARS.ctrl_arr_dpt_clipped)

ctrl_limits = {}
ctrl_limits['cmin'] = {}
ctrl_limits['cmax'] = {}

for i in range(cmax.shape[1]-4):
    ctrl_limits['cmin'][f'c1_{i}'] = cmin[0, i]
    ctrl_limits['cmin'][f'c3_{i}'] = cmin[1, i]
    ctrl_limits['cmin'][f'c5_{i}'] = cmin[2, i]
    ctrl_limits['cmax'][f'c1_{i}'] = cmax[0, i]
    ctrl_limits['cmax'][f'c3_{i}'] = cmax[1, i]
    ctrl_limits['cmax'][f'c5_{i}'] = cmax[2, i]


def model():
    # setting min and max value to be 0.1*true and 3.*true
    c1_list = []
    c3_list = []
    c5_list = []

    for i in range(cmax.shape[1]-4):
        # c1_list.append(numpyro.sample(f'c1_{i}', dist.Uniform(cmin[0, i], cmax[0, i])))
        c1_list.append(ctrl_arr_dpt[0, i])
        if i == 16:
            c3_list.append(numpyro.sample(f'c3_{i}',
                                          dist.Uniform(cmin[1, i], cmax[1, i])))
            c5_list.append(numpyro.sample(f'c5_{i}',
                                          dist.Uniform(cmin[2, i], cmax[2, i])))
        else:
            c3_list.append(ctrl_arr_dpt[1, i])
            c5_list.append(ctrl_arr_dpt[2, i])

    for i in range(4):
        c1_list.append(0.0)
        c3_list.append(0.0)
        c5_list.append(0.0)

    ctrl_arr = [jnp.array(c1_list),
                jnp.array(c3_list),
                jnp.array(c5_list)]

    eig_sample = jnp.array([])

    # building the entire hypermatrix
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  ctrl_arr, nc, len_s)
    
    delta_omega = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6
    delta_omega = delta_omega - eigvals_model
    delta_omega /= eigvals_sigma

    
    return numpyro.factor('obs',
                          dist.Normal(delta_omega, jnp.ones_like(delta_omega)).\
                          log_prob(jnp.zeros_like(delta_omega)))
    

def get_delta_omega():
    cdpt = GVARS.ctrl_arr_dpt_clipped

    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                fixed_hypmat_all_sparse,
                                                cdpt, nc, len_s)

    delta_omega = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6
    delta_omega -= eigvals_model
    # misfit = -0.5*np.sum(delta_omega**2)
    return delta_omega


def get_posterior_grid():
    N = 100
    fac = jnp.linspace(0.01, 2., N)
    misfit_model_arr = jnp.zeros(N)
    misfit_obs_arr = jnp.zeros(N)
    def true_func(ic, misfits):
        misfit_mod_arr, misfit_obs_arr = misfits
        cdpt = GVARS.ctrl_arr_dpt_clipped*1.0
        cdpt = jidx_update(cdpt, jidx[1, 2], GVARS.ctrl_arr_dpt_clipped[1, 2] * fac[ic])

        diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  cdpt, nc, len_s)
        delta_omega = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6
        delta_omega_model = delta_omega - eigvals_model
        delta_omega_obs = delta_omega - eigvals_true
        delta_omega_model /= eigvals_sigma
        delta_omega_obs /= eigvals_sigma
        misfit_mod = -0.5*np.sum(delta_omega_model**2)/len_data # + log_prob_shift
        misfit_obs = -0.5*np.sum(delta_omega_obs**2)/len_data # + log_prob_shift
        misfit_mod_arr = jidx_update(misfit_mod_arr, jidx[ic], misfit_mod)
        misfit_obs_arr = jidx_update(misfit_obs_arr, jidx[ic], misfit_obs)
        return (misfit_mod_arr, misfit_obs_arr)
    return fac, foril(0, N, true_func, (misfit_model_arr, misfit_obs_arr))

"""
with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
    theta = numpyro.sample('theta',
    dist.TransformedDistribution(dist.Normal(0., 1.),
                                 dist.transforms.AffineTransform(mu, tau)))
numpyro.sample('obs', dist.Normal(theta, sigma), obs=y) 
"""

# Start from this source of randomness. We will split keys for subsequent operations.
seed = int(ARGS.lmin + ARGS.lmax + ARGS.chain_num + int(100*np.random.rand()))
print(f"seed = {seed}")
rng_key = random.PRNGKey(seed)
rng_key, rng_key_ = random.split(rng_key)

"""
# Run NUTS.
#kernel = NUTS(model)
kernel = SA(model)#, init_strategy=init_to_value(values=ctrl_arr_dpt))
mcmc = MCMC(kernel, num_warmup=5500, num_samples=ARGS.maxiter)
mcmc.run(rng_key_, extra_fields=('potential_energy',))
pe = mcmc.get_extra_fields()['potential_energy']

metadata = {}
metadata['n0'] = ARGS.n0
metadata['lmin'] = ARGS.lmin
metadata['lmax'] = ARGS.lmax
metadata['rth'] = GVARS.rth
metadata['knot_num'] = GVARS.knot_num
metadata['maxiter'] = ARGS.maxiter

output_data = {}
output_data['samples'] = mcmc.get_samples()
output_data['metadata'] = metadata
output_data['ctrl_limits'] = ctrl_limits
output_data['potential_energy'] = pe

fname = f"output-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
jf.save_obj(output_data, f"{GVARS_PATHS.scratch_dir}/{fname}")

"""
_get_posterior_grid = jax.jit(get_posterior_grid)
fac, misfits = get_posterior_grid()
print(f"delta_omega = {get_delta_omega()}")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axs[0].plot(fac, misfits[0], 'k', label='model')
#axs[0].plot(fac, misfits[1], '--r', label='observed')
axs[0].legend()

axs[1].plot(fac, np.exp(misfits[0]), 'k', label='model')
#axs[1].plot(fac, np.exp(misfits[1]), '--r', label='observed')
axs[1].legend()
plt.savefig('model_vs_data_minfunc.pdf')


#--------- SECTION TO STORE MATRICES FOR COLLAB PROBLEM ------------#
c_fixed = np.zeros_like(GVARS.ctrl_arr_dpt_clipped)
c_fixed = GVARS.ctrl_arr_dpt_clipped.copy()
# for n0 = 0, lmin = lmax = 200, knots = 5, rth = 0.95
# the c3[2] and c5[2] are the most sensitive (sharpest gaussians)
# excluding their contribution in the total fixed part
c_fixed[1, 2] = 0.0
c_fixed[2, 2] = 0.0

noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
                                        precompute.build_hypmat_all_cenmults()

# this is the fixed part of the diag
diag_evals_fixed = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                    fixed_hypmat_all_sparse,
                                                    c_fixed, nc, len_s).todense()
diag_evals_fixed *= 1./2./omega0_arr*GVARS.OM*1e6

# we just need to save the noc_diag corresponding to the two ctrl_pts set to zero
noc_diag = [noc_hypmat_all_sparse[1][2].todense()/2./omega0_arr*GVARS.OM*1e6]
noc_diag.append(noc_hypmat_all_sparse[2][2].todense()/2./omega0_arr*GVARS.OM*1e6)


# checking if the forward problem works with the above components
pred = diag_evals_fixed + GVARS.ctrl_arr_dpt_clipped[1, 2] * noc_diag[0]\
       + GVARS.ctrl_arr_dpt_clipped[2, 2] * noc_diag[1]

true_params = np.array([GVARS.ctrl_arr_dpt_clipped[1,2],
                        GVARS.ctrl_arr_dpt_clipped[2,2]])

print('Pred - Data:\n', pred - eigvals_model)

np.save('fixed_part.npy', diag_evals_fixed)
np.save('param_coeff.npy', noc_diag)
np.save('data.npy', eigvals_model)
np.save('true_params.npy', true_params)


