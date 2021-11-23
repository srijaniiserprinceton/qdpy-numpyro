import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import arviz as az
import sys

def log_likelihood(theta, y, yerr):
    itereigs = model(theta)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - itereigs) ** 2 / sigma2)

def log_prior(theta):
    tval = True
    for i in range(num_params):
        if 0.05 < theta[i] < 1.95:
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
    c3 = theta[:nc] * true_params[0, :]
    c5 = theta[nc:] * true_params[1, :]
    pred = fixed_part + c3 @ param_coeff[0] + c5 @ param_coeff[1]
    return pred


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
    # loading the files forthe problem
    data = np.load('data.npy')
    true_params = np.load('true_params.npy')
    param_coeff = np.load('param_coeff.npy')
    fixed_part = np.load('fixed_part.npy')
    cind_arr = np.load('cind_arr.npy')
    check_loaded_data()

    # number of s and c to fit
    len_s = true_params.shape[0]
    nc = true_params.shape[1]
    num_params = len(cind_arr)
    var_params = np.ones_like(true_params).flatten()

    # putting the true params
    refs = {}
    # initializing the keys
    for sind in range(1, 3):
        s = 2*sind + 1
        for ci in range(num_params):
            refs[f"c{s}_{ci}"] = true_params[sind-1, ci]

    pos = var_params.flatten() * (1 + 0.1* np.random.randn(750, len_s*nc))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=(data, 1.0))
    sampler.run_mcmc(pos, 5250, progress=True);

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

