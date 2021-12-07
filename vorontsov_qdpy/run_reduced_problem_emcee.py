import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import emcee
import corner
import sys

def log_likelihood(theta, y, yerr):
    itereigs = model(theta)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - itereigs) ** 2 / sigma2)

def log_prior(theta):
    tval = True
    for i in range(num_params):
        if 0.90 < theta[i] < 1.10:
            tval *= True
        else:
            return -np.inf
    return 0.0


def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)


def model(theta):
    c_arr = theta * true_params
    pred = fixed_part + c_arr @ param_coeff

    acoeff_model = np.array([])

    for i in range(nmults):
        ell0 = ell0_arr[i]
        omegaref = omega0_arr[i]
        pred_dense = sparse.coo_matrix((pred[i], (sparse_idx[i, :, 0],
                                                  sparse_idx[i, :, 1])),
                                        shape=(dim_hyper, dim_hyper))
        eigval_mult = get_eigs(pred_dense.toarray())/2./omegaref*OMval*1e6
        pred_acoeff = (Pjl[i] @ eigval_mult)/Pjl_norm[i]
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

    # loading the files forthe problem
    dim_hyper = int(np.loadtxt('.dimhyper'))

    # true params from Antia wsr
    true_params = np.load('true_params.npy')
    eigvals_true = np.load('evals_model.npy')
    acoeffs_true = np.load('acoeffs_true.npy')
    acoeffs_sigma = np.load('acoeffs_sigma.npy')

    ell0_arr = np.load('ell0_arr.npy')
    omega0_arr = np.load('omega0_arr.npy')
    nmults = len(ell0_arr)

    cind_arr = np.load('cind_arr.npy')
    smin_ind, smax_ind = np.load('sind_arr.npy')

    param_coeff = np.load('param_coeff.npy')
    sparse_idx = np.load('sparse_idx.npy')
    fixed_part = np.load('fixed_part.npy')
    param_coeff = param_coeff[smin_ind:smax_ind+1, ...]

    # number of s and c's to fit
    len_s = true_params.shape[0]
    nc = true_params.shape[1]

    # reshaping and moving axis to allow seamless np.dot
    true_params = np.reshape(true_params, (nc * len_s,), 'F')
    param_coeff = np.reshape(param_coeff, (nc * len_s, nmults, -1), 'F')
    param_coeff = np.moveaxis(param_coeff, 0, 1)
    num_params = len(true_params)

    # setting the prior limits
    cmin = np.ones_like(true_params) * 0.9
    cmax = np.ones_like(true_params) * 1.1

    # Reading RL poly from precomputed file
    # shape (nmults x (smax+1) x 2*ellmax+1)
    # reshaping to (nmults x (smax+1) x dim_hyper)
    RL_poly = np.load('RL_poly.npy')
    s_arr = np.array([1, 3, 5])
    smin = min(s_arr)
    smax = max(s_arr)
    Pjl_read = RL_poly[:, smin:smax+1:2, :]
    Pjl = np.zeros((Pjl_read.shape[0],
                    Pjl_read.shape[1],
                    dim_hyper))
    Pjl[:, :, :Pjl_read.shape[2]] = Pjl_read

    # calculating the normalization for Pjl apriori
    # shape (nmults, num_j)
    Pjl_norm = np.zeros((nmults, Pjl.shape[1]))
    for mult_ind in range(nmults):
        Pjl_norm[mult_ind] = np.diag(Pjl[mult_ind] @ Pjl[mult_ind].T)

    # number of s and c to fit
    var_params = np.ones_like(true_params)

    pos = var_params.flatten() * (1 + 0.01* np.random.randn(25, len_s*nc))
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

