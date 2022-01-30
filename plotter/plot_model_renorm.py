import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_model(params, len_s, suptitle, figname):
    num_params = len(params)
    fig, ax = plt.subplots(nrows=len_s, ncols=3, figsize=(10, 8))
    ax_flat = ax.flatten()
    chosen_indices = np.array([0, num_params//2, -2])
    imax = len(chosen_indices)
    
    for i in range(len(chosen_indices)):
        for j in range(len_s):
            param_ind = chosen_indices[i]
            data = params[param_ind+j]
            # the histogram of the data
            n, bins, patches = ax_flat[j*imax + i].hist(data, 40,
                                                        facecolor='green', alpha=0.75)
            ax_flat[j*imax + i].grid(True)
    plt.tight_layout()
    fig.suptitle(suptitle, size=16)
    fig.subplots_adjust(top=0.92)
    plt.savefig(f'{figname}.png')
    return None


def visualize_model_renorm(true_params_flat, true_params_samples, sigma,
                           renorm_fn, len_s):
    num_params = len(true_params_flat)
    
    # to facilitate fivision without using NAX
    true_params_flat_shaped = np.reshape(true_params_flat, (num_params, 1))
    sigma = np.reshape(sigma, (num_params, 1))

    plot_model(true_params_samples, len_s, 'Crude $m$', 'crude_params')
    
    true_params_samples_renormed = renorm_fn(true_params_samples,
                                             true_params_flat_shaped,
                                             1.0)
    plot_model(true_params_samples_renormed, len_s,
               'Step 1 normalization: $\\frac{m-m_0}{m_0}$',
               'renormed_params')

    true_params_samples_final = true_params_samples_renormed/sigma
    plot_model(true_params_samples_final, len_s,
               'Step 2 normalization: $\\frac{m-m_0}{m_0}/\sigma$',
               'final_params')
