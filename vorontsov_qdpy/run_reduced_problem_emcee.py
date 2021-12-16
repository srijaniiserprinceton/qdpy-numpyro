import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.integrate import simps
import emcee
import corner
import sys

from qdpy_jax import globalvars as gvar_jax
NAX = np.newaxis


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


def log_likelihood(theta, y, yerr):
    itereigs = model(theta)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - itereigs) ** 2 / sigma2)


def log_prior(theta):
    tval = True
    for i in range(num_params):
        if 0.80 < theta[i] < 1.20:
            tval *= True
        else:
            return -np.inf
    return 0.0


def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)


def get_clp(bkm):
    tvals = np.linspace(0, np.pi, 25)
    integrand = np.zeros((bkm.shape[0],
                          bkm.shape[1],
                          bkm.shape[-1],
                          len(tvals)))

    for i in range(len(tvals)):
        term2 = 2*bkm*np.sin(k_arr*tvals[i])/k_arr_denom
        term2 = term2.sum(axis=(1, 2))[:, NAX, :]
        integrand[:, :, :, i] = np.cos(p_arr[:, :, NAX]*tvals[i] - term2)

    integral = simps(integrand, axis=-1, x=tvals)/np.pi
    return integral


def get_eig_corr(clp, z1):
    cZc = clp.conj() * ((z1 * clp[:, NAX, :]).sum(axis=0))
    return cZc.sum(axis=0)



def model(theta):
    c_params = theta * true_params
    z0 = param_coeff_M @ c_params + fixed_part_M
    zfull = param_coeff @ c_params + fixed_part
    bkm = param_coeff_bkm @ c_params + fixed_part_bkm
    for i in range(nmults):
        bkm[i] = bkm[i]*(-1.0)/dom_dell_arr[i]
    clp = get_clp(bkm)

    acoeff_model = np.array([])

    for i in range(nmults):
        omegaref = omega0_arr[i]
        ell0 = ell0_arr[i]

        z0mult = z0[i]
        z1mult = zfull[i]/2./omegaref - z0mult
        _eigval0mult = get_eig_corr(clp[mult_ind], z0mult)*GVARS.OM*1e6
        _eigval1mult = get_eig_corr(clp[mult_ind], z1mult)*GVARS.OM*1e6
        _eigval_mult = _eigval0mult + _eigval1mult

        pred_acoeff = (Pjl[i] @ _eigval_mult)/Pjl_norm[i]
        acoeff_model = np.append(acoeff_model, pred_acoeff)

    return acoeff_model


def eigval_sort_slice(eigval, eigvec):
    eigbasis_sort = np.zeros(len(eigval), dtype=int)

    for i in range(len(eigval)):
        eigbasis_sort[i] = np.argmax(abs(eigvec[i]))

    return eigval[eigbasis_sort]


def get_eigs(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = eigval_sort_slice(eigvals, eigvecs)
    return eigvals


def print_summary(samples, ctrl_arr):
    keys = samples.keys()
    for key in keys:
        sample = samples[key]
        key_split = key.split("_")
        idx = int(key_split[-1])
        sidx = int((int(key_split[0][1])-1)//2)
        obs = ctrl_arr[sidx-1, idx] / 1e-3
        print(f"[{obs:11.4e}] {key}: {sample.mean():.4e} +/- {sample.std():.4e}:" +
              f"error/sigma = {(sample.mean()-obs)/sample.std():8.3f}")
    return None


def check_loaded_data():
    # checking that the loaded data are correct
    pred = fixed_part * 1.0

    # adding the contribution from the fitting part
    for sind in range(1,3):
        for ci, cind in enumerate(cind_arr):
            pred += true_params[sind-1, ci] * param_coeff[sind-1][ci]

    # these arrays should be very close
    np.testing.assert_array_almost_equal(pred, data)
    return None


if __name__ == "__main__":
    OMval = 2.0963670602632024e-05


    ############################################################################
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
    k_arr_denom[k_arr==0] = 1


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
    Pjl[:, :, :Pjl_read.shape[2]] = Pjl_read

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


    # reshaping and moving axis to allow seamless np.dot
    true_params = np.reshape(true_params, (nc * len_s,), 'F')
    param_coeff = np.reshape(param_coeff, (nc * len_s, nmults,
                                           max_nbs, max_nbs, len_mmax), 'F')
    param_coeff_M = np.reshape(param_coeff_M, (nc * len_s, nmults,
                                               max_nbs, max_nbs, len_mmax), 'F')
    param_coeff_bkm = np.reshape(param_coeff_bkm, (nc * len_s, nmults,
                                                   max_nbs, max_nbs, len_mmax), 'F')

    # moving axis to allow seamless np.dot
    param_coeff = np.moveaxis(param_coeff, 0, -1)
    param_coeff_M = np.moveaxis(param_coeff_M, 0, -1)
    param_coeff_bkm = np.moveaxis(param_coeff_bkm, 0, -1)

    # setting the prior limits
    cmin = 0.8 * np.ones_like(true_params)
    cmax = 1.2 * np.ones_like(true_params)

    ############################################################################
    # number of s and c to fit
    var_params = np.ones_like(true_params)

    pos = var_params.flatten() * (1 + 0.01* np.random.randn(60, len_s*nc))
    nwalkers, ndim = pos.shape
    # sys.exit()

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=(acoeffs_true, acoeffs_sigma))
    sampler.run_mcmc(pos, 525, progress=True);

    fig1, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        # ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    fig1.show()

    flat_samples = sampler.get_chain(discard=25, thin=15, flat=True)

    fig2 = corner.corner(flat_samples,
                        truths=[tval for tval in var_params.flatten()])
    fig2.show()

