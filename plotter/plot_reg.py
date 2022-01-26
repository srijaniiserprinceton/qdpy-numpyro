import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rescale(a):
    return (a - a.mean())/(a.max() - a.min())

def get_slope(y, x):
    return (y[1:] - y[:-1])/(x[1:] - x[:-1])

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/dpy_jax"

reg_data = pd.read_csv(f"{package_dir}/dpy_jax/reg_misfit_jesper_360d.txt")

mask = (reg_data['mu'] > 1e-5) * (reg_data['mu'] < 1e10)
data_misfit = reg_data['data-misfit'][mask].values
model_misfit = reg_data['model-misfit'][mask].values
mu = reg_data['mu'][mask].values

data_mf_rescaled = rescale(data_misfit)
model_mf_rescaled = rescale(model_misfit)
slope = get_slope(model_mf_rescaled, data_mf_rescaled)
knee_idx = np.argmin(abs(slope+1))

range_x = data_misfit.max() - data_misfit.min()
range_y = model_misfit.max() - model_misfit.min()

max_nbs = 20
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axs = axs.flatten()
axs[0].plot(data_misfit, model_misfit, '+k')
for i in range(2*max_nbs):
    axs[0].plot(data_misfit[knee_idx-max_nbs+i],
                model_misfit[knee_idx-max_nbs+i], '+r')
axs[0].plot(data_misfit[knee_idx], model_misfit[knee_idx], marker='s', color='red')
axs[0].text(data_misfit[knee_idx]+0.04*range_x, 
            model_misfit[knee_idx]+0.04*range_y,
            f"$\\mu$ = {mu[knee_idx]:.5f}")
axs[0].set_xlabel('Data misfit')
axs[0].set_ylabel('Roughness')
axs[0].set_title('L-curve')

axs[1].hist(mu[knee_idx-max_nbs:knee_idx+max_nbs])
axs[1].set_xlabel('Regularization parameter $\\mu$')
axs[1].set_ylabel('Count')
axs[1].set_title('Distribution of $\\mu$ in the L-curve knee region')
fig.tight_layout()
plt.show(fig)
