BBBBimport os
import time
num_chains = 1
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_chains}"

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import argparse
import sys

import jax
import jax.numpy as jnp
from jax import random
from jax.config import config
from jax.experimental import sparse
from jax.lax import fori_loop as foril
from jax.lax import dynamic_slice as jdc
from jax.lax import dynamic_update_slice as jdc_update
from jax.ops import index_update as jidx_update
from jax.ops import index as jidx
config.update('jax_enable_x64', True)

NAX = jnp.newaxis

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SA

print(jax.devices()) # printing JAX devices
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import jax_functions as jf
from qdpy_jax import build_hypermatrix_sparse as build_hm_sparse

parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
parser.add_argument("--chain_num", help="chain number",
                    type=int, default=1)
parser.add_argument("--warmup", help="number of warmup steps",
                    type=int, default=20)
PARSED_ARGS = parser.parse_args()

read_params = np.loadtxt(".n0-lmin-lmax.dat")
ARGS = {}
ARGS['n0'] = int(read_params[0])
ARGS['lmin'] = int(read_params[1])
ARGS['lmax'] = int(read_params[2])
ARGS['rth'] = read_params[3]
ARGS['knot_num'] = int(read_params[4])
ARGS['load_from_file'] = int(read_params[5])

GVARS = gvar_jax.GlobalVars(n0=ARGS['n0'],
                            lmin=ARGS['lmin'],
                            lmax=ARGS['lmax'],
                            rth=ARGS['rth'],
                            knot_num=ARGS['knot_num'],
                            load_from_file=ARGS['load_from_file'])

################ loading the files for the problem ###############
dim_hyper = int(np.loadtxt('.dimhyper'))

# true params from Antia wsr
true_params = np.load('true_params.npy')
eigvals_true = np.load('evals_model.npy')
acoeffs_true = np.load('acoeffs_true.npy')
acoeffs_sigma = np.load('acoeffs_sigma.npy')

ell0_arr = np.load('ell0_arr.npy')
omega0_arr = np.load('omega0_arr.npy')
dom_dell_arr = np.load('dom_dell_arr.npy').flatten()

cind_arr = np.load('cind_arr.npy')
smin_ind, smax_ind = np.load('sind_arr.npy')

# supermatrix specific files for the exact problem
param_coeff = np.load('param_coeff.npy')
sparse_idx = np.load('sparse_idx.npy')
fixed_part = np.load('fixed_part.npy')
param_coeff = param_coeff[smin_ind:smax_ind+1, ...]
max_nbs = param_coeff.shape[-2]
len_mmax = param_coeff.shape[-1]
max_lmax = int(len_mmax//2)

# supermatrix M specific files for the V11 approximated problem
param_coeff_M = np.load('param_coeff_M.npy')
sparse_idx_M = np.load('sparse_idx_M.npy')
fixed_part_M = np.load('fixed_part_M.npy')
param_coeff_M = param_coeff_M[smin_ind:smax_ind+1, ...]

# reading the bkm values
fixed_part_bkm = np.load('fixed_bkm.npy')
param_coeff_bkm = np.load('noc_bkm.npy')
param_coeff_bkm = param_coeff_bkm[smin_ind:smax_ind+1, ...]
k_arr = np.load('k_arr.npy')
p_arr = np.load('p_arr.npy')

k_arr_denom = k_arr*1
k_arr_denom[k_arr==0] = np.inf

#################################################################
# number of central multiplets
nmults = len(GVARS.ell0_arr)
# total number of s for which to create a-coeffs
num_j = len(GVARS.s_arr)
# number of s to fit
len_s = true_params.shape[0]
# number of c's to fit
nc = true_params.shape[1]
num_params = len(cind_arr)
num_k = (np.unique(k_arr)>0).sum()

#############################################################
# Reading RL poly from precomputed file
# shape (nmults x (smax+1) x 2*ellmax+1)
# reshaping to (nmults x (smax+1) x dim_hyper)
RL_poly = np.load('RL_poly.npy')
smin = min(GVARS.s_arr)
smax = max(GVARS.s_arr)
Pjl_read = RL_poly[:, smin:smax+1:2, :]
Pjl = np.zeros((Pjl_read.shape[0],
                Pjl_read.shape[1],
                len_mmax))
Pjl_hyper = np.zeros((Pjl_read.shape[0],
                      Pjl_read.shape[1],
                      dim_hyper))
Pjl[:, :, :Pjl_read.shape[2]] = Pjl_read

for i in range(nmults):
    ell0 = ell0_arr[i]
    Pjl_hyper[i, :, :2*ell0+1] = Pjl_read[i, :, :2*ell0+1]

# calculating the normalization for Pjl apriori
# shape (nmults, num_j)
Pjl_norm = np.zeros((nmults, Pjl.shape[1]))
for mult_ind in range(nmults):
    Pjl_norm[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

ellmax_pjl = int(Pjl_read.shape[-1]//2)
dell = max_lmax - ellmax_pjl
Pjl = np.array(Pjl)
for i in range(nmults):
    dell2 = ellmax_pjl - ell0_arr[i]
    Pjl[i, ...] = np.roll(Pjl[i, ...], dell+dell2, axis=-1)

#############################################################
# converting to device array
Pjl = jnp.asarray(Pjl)
Pjl_norm = jnp.asarray(Pjl_norm)
Pjl_hyper = jnp.asarray(Pjl_hyper)

acoeffs_sigma = jnp.asarray(acoeffs_sigma)
true_params = jnp.asarray(true_params)
param_coeff = jnp.asarray(param_coeff)
fixed_part = jnp.asarray(fixed_part)

param_coeff_M = jnp.asarray(param_coeff_M)
fixed_part_M = jnp.asarray(fixed_part_M)

param_coeff_bkm = jnp.asarray(param_coeff_bkm)
fixed_part_bkm = jnp.asarray(fixed_part_bkm)
k_arr = jnp.asarray(k_arr)

sparse_idx = jnp.asarray(sparse_idx)
ell0_arr_jax = jnp.asarray(GVARS.ell0_arr)
omega0_arr_jax = jnp.asarray(omega0_arr)
dom_dell_jax = jnp.asarray(dom_dell_arr)

########################################################################
# reshaping true_params and param_coeff

true_params = jnp.reshape(true_params, (nc * len_s,), 'F')
param_coeff = jnp.reshape(param_coeff, (nc * len_s, nmults,
                                        max_nbs, max_nbs, len_mmax), 'F')
param_coeff_M = jnp.reshape(param_coeff_M, (nc * len_s, nmults,
                                            max_nbs, max_nbs, len_mmax), 'F')
param_coeff_bkm = jnp.reshape(param_coeff_bkm, (nc * len_s, nmults,
                                                max_nbs, max_nbs, len_mmax), 'F')

# moving axis to allow seamless jnp.dot
param_coeff = jnp.moveaxis(param_coeff, 0, -1)
param_coeff_M = jnp.moveaxis(param_coeff_M, 0, -1)
param_coeff_bkm = jnp.moveaxis(param_coeff_bkm, 0, -1)

# setting the prior limits
cmin = 0.8 * jnp.ones_like(true_params)
cmax = 1.2 * jnp.ones_like(true_params)

ctrl_limits = {}
ctrl_limits['cmin'] = cmin
ctrl_limits['cmax'] = cmax

# comparing the matrices with the true values
supmat_jax = param_coeff @ true_params + fixed_part

def get_clp(bkm):
    tvals = jnp.linspace(0, jnp.pi, 25)
    
    # integrand of shape (ell, p, m ,t)
    integrand = jnp.zeros((p_arr.shape[0],
                           p_arr.shape[1],
                           p_arr.shape[2],
                           len(tvals)))

    def t_func(i, intg):
        term2 = 2. * bkm * jnp.sin(k_arr*tvals[i]) / k_arr_denom
        term2 = term2.sum(axis=1)
        intg = jidx_update(intg,
                           jidx[:, :, :, i],
                           jnp.cos(p_arr * tvals[i] - term2[:, NAX, :]))
        # intg = foril(0, max_nbs, p_func, intg)
        return intg

    integrand = foril(0, len(tvals), t_func, integrand)
    integral = jnp.trapz(integrand, axis=-1, x=tvals)/jnp.pi
    return integral


def get_eig_corr(clp, z1):
    # cZc = jnp.zeros(z1.shape[-1])
    # def m_func(idx, czcm):
    #     czcm = jidx_update(czcm,
    #                        jidx[idx],
    #                        clp.conj()[:, idx] @ (z1[:, :, idx] @ clp[:, idx]))
    #     return czcm
    # return foril(0, len(cZc), m_func, cZc)
    return (clp.conj() * ((z1 * clp[:, NAX, :]).sum(axis=0))).sum(axis=0)


def compare_model():
    # predicted a-coeficients
    eigvals_compute = jnp.array([])
    eigvals_acoeffs = jnp.array([])
    pred_acoeffs = jnp.zeros(num_j * nmults)
    pred_ac_dpt = jnp.zeros(num_j * nmults)

    def scale_bkm(mult_idx, bkm_full):
        bkm_full = jidx_update(bkm_full,
                               jidx[i, :, :, :],
                               -1.0*bkm_full[mult_idx, ...]/dom_dell_jax[mult_idx])
        return bkm_full

    z0 = param_coeff_M @ true_params + fixed_part_M
    zfull = param_coeff @ true_params + fixed_part
    bkm = param_coeff_bkm @ true_params + fixed_part_bkm
    bkm = foril(0, nmults, scale_bkm, bkm)
    clp = get_clp(bkm)

    def loop_in_mults(mult_ind, pred_acoeff):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]

        z0mult = z0[mult_ind]
        z1mult = zfull[mult_ind]/2./omegaref - z0mult

        _eigval0mult = get_eig_corr(clp[mult_ind], z0mult)*GVARS.OM*1e6
        _eigval1mult = get_eig_corr(clp[mult_ind], z1mult)*GVARS.OM*1e6
        _eigval_mult = _eigval0mult + _eigval1mult

        Pjl_local = Pjl[mult_ind]
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    def loop_dpt(mult_ind, pred_ac_dpt):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]
        zmult = zfull[mult_ind]/2./omegaref
        zm_flat = jnp.reshape(zmult, (max_nbs*max_nbs*len_mmax), order='F')
        sidx_flat = jnp.reshape(sparse_idx[mult_ind],
                                (max_nbs*max_nbs*len_mmax, 2), order='F')
        zmult_dense = sparse.BCOO((zm_flat, sidx_flat),
                                  shape=(dim_hyper, dim_hyper)).todense()
        _eigvalmult = jnp.diag(zmult_dense)*GVARS.OM*1e6
        Pjl_local = Pjl_hyper[mult_ind]
        pred_ac_dpt = jdc_update(pred_ac_dpt,
                                 (Pjl_local @ _eigvalmult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_ac_dpt

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    pred_ac_dpt = foril(0, nmults, loop_dpt, pred_ac_dpt)
    return pred_acoeffs, pred_ac_dpt

compare_model_ = jax.jit(compare_model)

def model():
    # c_arr = numpyro.sample(f'c_arr', dist.Uniform(cmin, cmax))
    c_arr = jnp.ones_like(true_params)
    pred_acoeffs = jnp.zeros(num_j * nmults)
    c_params = c_arr * true_params

    def scale_bkm(mult_idx, bkm_full):
        bkm_full = jidx_update(bkm_full,
                               jidx[i, :, :, :],
                               -1.0*bkm_full[mult_idx, ...]/dom_dell_jax[mult_idx])
        return bkm_full

    zfull = param_coeff @ c_params + fixed_part
    bkm = param_coeff_bkm @ c_params + fixed_part_bkm
    bkm = foril(0, nmults, scale_bkm, bkm)
    clp = get_clp(bkm)

    def loop_in_mults(mult_ind, pred_acoeff):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]
        z_mult = zfull[mult_ind]/2./omegaref
        _eigval_mult = get_eig_corr(clp[mult_ind],
                                    z_mult)*GVARS.OM*1e6

        Pjl_local = Pjl[mult_ind]
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    # misfit_acoeffs = (pred_acoeffs - acoeffs_true)/acoeffs_sigma
    # return numpyro.factor('obs', dist.Normal(0.0, 1.0).log_prob(misfit_acoeffs))
    return pred_acoeffs

def model_old():
    c_arr = numpyro.sample(f'c_arr', dist.Uniform(cmin, cmax))
    pred_acoeffs = jnp.zeros(num_j * nmults)
    c_params = c_arr * true_params

    def scale_bkm(mult_idx, bkm_full):
        bkm_full = jidx_update(bkm_full,
                               jidx[i, :, :, :],
                               -1.0*bkm_full[mult_idx, ...]/dom_dell_jax[mult_idx])
        return bkm_full

    z0 = param_coeff_M @ c_params + fixed_part_M
    zfull = param_coeff @ c_params + fixed_part
    bkm = param_coeff_bkm @ c_params + fixed_part_bkm
    bkm = foril(0, nmults, scale_bkm, bkm)
    clp = get_clp(bkm)

    def loop_in_mults(mult_ind, pred_acoeff):
        ell0 = ell0_arr_jax[mult_ind]
        omegaref = omega0_arr_jax[mult_ind]

        z0mult = z0[mult_ind]
        z1mult = zfull[mult_ind]/2./omegaref - z0mult

        _eigval0mult = get_eig_corr(clp[mult_ind], z0mult)*GVARS.OM*1e6
        _eigval1mult = get_eig_corr(clp[mult_ind], z1mult)*GVARS.OM*1e6
        _eigval_mult = _eigval0mult + _eigval1mult

        Pjl_local = Pjl[mult_ind]
        pred_acoeff = jdc_update(pred_acoeff,
                                 (Pjl_local @ _eigval_mult)/Pjl_norm[mult_ind],
                                 (mult_ind * num_j,))
        return pred_acoeff

    pred_acoeffs = foril(0, nmults, loop_in_mults, pred_acoeffs)
    misfit_acoeffs = (pred_acoeffs - acoeffs_true)/acoeffs_sigma
    return numpyro.factor('obs', dist.Normal(0.0, 1.0).log_prob(misfit_acoeffs))


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


def print_summary(samples, ctrl_arr):
    count = 0
    for i in range(len(true_params)):
        sample = samples['c_arr'][:, i] * true_params[i]
        obs = true_params[i] 
        print(f"[{obs:11.4e}] c_arr[{i}]: {sample.mean():.4e} +/- {sample.std():.4e}:" +
              f"error/sigma = {(sample.mean()-obs)/sample.std():8.3f}")
    return None


def test_setup():
    acoeffs_model, acoeffs_dpt = compare_model()
    diff = acoeffs_dpt - acoeffs_true
    len_acoeffs = len(acoeffs_model)
    print(f"{'ell':^6}   |" +
          f"{'Model':^12}   |" +
          f"{'True':^12}   |" +
          f"{'Diff':^12}   |" +
          f"{'Diff/sigma':^12}\n" + 
          f"-----------------------------------------------------------------------")
    mult_idx = 0
    for i in range(len_acoeffs):
        print(f"{ell0_arr_jax[mult_idx]:^6}   |" +
              f"{acoeffs_dpt[i]*1e3:12.5f}   |" +
              f"{acoeffs_true[i]*1e3:12.5f}   |" +
              f"{diff[i]*1e3:12.5f}   |" +
              f"{diff[i]/acoeffs_sigma[i]:12.5f}")
        if (i+1)%3==0:
          print(f"-------------------------------------------------------------------")
          mult_idx += 1

    t1 = time.time()
    N = 50
    count = 0
    for i in range(N):
        _temp = compare_model_()
        count += _temp.sum()/N
    t2 = time.time()
    print(f"Total time taken for {nmults} modes = {(t2-t1)/N:.3e} seconds")


sys.exit()

if __name__ == "__main__":
    test_setup()

    # Start from this source of randomness. We will split keys for subsequent operations.
    seed = int(123 + 100*np.random.rand())
    rng_key = random.PRNGKey(seed)
    rng_key, rng_key_ = random.split(rng_key)
    # sys.exit()

    kernel = NUTS(model, max_tree_depth=(5, 4),
                  find_heuristic_step_size=True)
    mcmc = MCMC(kernel,
                num_warmup=PARSED_ARGS.warmup,
                num_samples=PARSED_ARGS.maxiter,
                num_chains=num_chains)  
    mcmc.run(rng_key_, extra_fields=('potential_energy',))
    pe = mcmc.get_extra_fields()['potential_energy']

    # extracting necessary fields for plotting
    mcmc_sample = mcmc.get_samples()
    keys = mcmc_sample.keys()

    metadata = {}
    metadata['n0'] = ARGS['n0']
    metadata['lmin'] = ARGS['lmin']
    metadata['lmax'] = ARGS['lmax']
    metadata['rth'] = GVARS.rth
    metadata['knot_num'] = GVARS.knot_num
    metadata['maxiter'] = PARSED_ARGS.maxiter

    output_data = {}
    output_data['samples'] = mcmc.get_samples()
    output_data['metadata'] = metadata
    output_data['ctrl_limits'] = ctrl_limits

    fname = f"output-{PARSED_ARGS.maxiter}-{PARSED_ARGS.chain_num:03d}"
    jf.save_obj(output_data, f"{GVARS.scratch_dir}/{fname}")
    print_summary(mcmc_sample, true_params)

    plot_samples = {}

    # putting the true params
    refs = {}
    # initializing the keys
    for idx in range(num_params):
        sind = idx % 2
        ci = int(idx//2)
        s = 2*sind + 3
        argstr = f"c{s}_{ci}"
        refs[argstr] = true_params[idx]
        plot_samples[argstr] = output_data['samples']['c_arr'][:, idx]

    ax = az.plot_pair(
        plot_samples,
        var_names=[key for key in plot_samples.keys()],
        kde_kwargs={"fill_last": False},
        kind=["scatter", "kde"],
        marginals=True,
        point_estimate="median",
        figsize=(10, 8),
        reference_values=refs,
        reference_values_kwargs={'color':"red",
                                 "marker":"o",
                                 "markersize":6}
    )
    plt.tight_layout()
    plt.savefig(f'{GVARS.scratch_dir}/corner-reduced-{PARSED_ARGS.chain_num:03d}.png')

    sys.exit()
