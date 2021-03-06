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
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
plotdir = f"{scratch_dir}/plots"
knee_mu = np.array([ 0.1225026, 1.34397396, 11.34306532, 13.95488723, 200.00])
sval = 1
sind = int((sval+1)//2 - 1)

# suffix = "r09.w1.72d.gyre"
suffix = f"s{sval}"
reg_data = pd.read_csv(f"{package_dir}/dpy_jax/reg_misfit_{suffix}.txt")

data_misfit = reg_data['data-misfit']
model_misfit = reg_data['model-misfit']
mu = reg_data['mu']
sort_idx = np.argsort(mu.values)

mu = mu.values[sort_idx]
data_misfit = data_misfit.values[sort_idx]
model_misfit = model_misfit.values[sort_idx]

mask = (mu > 1e-16) * (mu < 1e15)
data_misfit = data_misfit[mask]
model_misfit = model_misfit[mask]
mu = mu[mask]

data_mf_rescaled = rescale(data_misfit)
model_mf_rescaled = rescale(model_misfit)
slope = get_slope(model_mf_rescaled, data_mf_rescaled)
knee_idx = np.argmin(abs(slope+1))
knee_idx = np.argmin(abs(mu - knee_mu[sind]))

range_x = model_misfit.max() - model_misfit.min()
range_y = data_misfit.max() - data_misfit.min()

max_nbs = 5
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axs = axs.flatten()
axs[0].semilogx(model_misfit, data_misfit, '+k')
for i in range(2*max_nbs):
    axs[0].loglog(model_misfit[knee_idx-max_nbs+i],
                  data_misfit[knee_idx-max_nbs+i], '+r')
axs[0].semilogx(model_misfit[knee_idx], data_misfit[knee_idx], marker='s', color='red')
axs[0].text(model_misfit[knee_idx],
            data_misfit[knee_idx]+0.04*range_y, 
            f"$\\mu$ = {mu[knee_idx]:.5e}")
axs[0].set_xlabel('Model roughness')
axs[0].set_ylabel('Data misfit')
axs[0].set_title('L-curve')

axs[1].hist(mu[knee_idx-max_nbs:knee_idx+max_nbs])
axs[1].set_xlabel('Regularization parameter $\\mu$')
axs[1].set_ylabel('Count')
axs[1].set_title('Distribution of $\\mu$ in the L-curve knee region')
fig.tight_layout()
fig.savefig(f"{plotdir}/regplot.{suffix}.png")
fig.show()
print(f"Saving plot to {plotdir}/regplot.{suffix}.png")

y = data_misfit
x = model_misfit
slope = np.diff(data_mf_rescaled)/np.diff(model_mf_rescaled)
plt.figure()
# plt.semilogy(model_misfit[1:], abs(slope))
plt.semilogy(abs(slope))
# plt.semilogy(model_misfit[knee_idx], abs(slope)[knee_idx], '*r')
plt.semilogy(knee_idx, abs(slope)[knee_idx], '*r')
plt.show()

# knee_idx2 = np.argmin(abs(slope+1e-12))
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# axs = axs.flatten()
# axs[0].plot(model_misfit, data_misfit, '+k')
# for i in range(2*max_nbs):
#     axs[0].plot(model_misfit[knee_idx2-max_nbs+i],
#                   data_misfit[knee_idx2-max_nbs+i], '+r')
# axs[0].plot(model_misfit[knee_idx2], data_misfit[knee_idx2],
#               marker='s', color='red')
# axs[0].text(model_misfit[knee_idx2]+0.04*range_x,
#             data_misfit[knee_idx2]+0.04*range_y, 
#             f"$\\mu$ = {mu[knee_idx]:.5e}")
# axs[0].set_xlabel('Roughness')
# axs[0].set_ylabel('Data misfit')
# axs[0].set_title('L-curve')
