import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from qdpy import jax_functions as jf


#-----------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--daynum", help="day from MDI epoch",
                    type=int, default=6328)
PARGS = parser.parse_args()
#---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
plotter_dir = os.path.dirname(os.path.dirname(current_dir))
package_dir = os.path.dirname(plotter_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

# loading the summaryfiles
# loading the final dpy fitted .npy file
try:
    summary = jf.load_obj(f'{scratch_dir}/hybrid-summary/summary_{PARGS.daynum}')
    summary_dpy = jf.load_obj(f'{scratch_dir}/dpy-summary/summary_{PARGS.daynum}')
    wsr_sigma = np.load(f'{scratch_dir}/sigma-files/wsr_sigma_{PARGS.daynum}.npy')
except FileNotFoundError:
    print(f"Files not found: Try running {package_dir}/organize/collect_output.py")
    sys.exit()

# loading the final fitted hybrid carr_fit from the summary files and the true params
true_params_flat = summary['true_params_flat']
carr_fit_hybrid = summary['c_arr_fit'] * true_params_flat
carr_fit_dpt = summary_dpy['c_arr_fit'] * true_params_flat

# making the wsr profiles
bsp_basis = summary['params']['dpy']['GVARS'].bsp_basis_full

# sind_arr = np.array([0,1,2])
# cind_arr = np.arange(len(carr_fit_dpt)//3)
sind_arr = np.arange(5)
cind_arr = np.arange(len(carr_fit_dpt)//5)

carr_fit_dpt_full = jf.c4fit_2_c4plot(summary['params']['dpy']['GVARS'],
                                      carr_fit_dpt, sind_arr, cind_arr)

carr_fit_hybrid_full = jf.c4fit_2_c4plot(summary['params']['dpy']['GVARS'],
                                         carr_fit_hybrid, sind_arr, cind_arr)

wsr_dpt_full = carr_fit_dpt_full @ bsp_basis
wsr_hybrid_full = carr_fit_hybrid_full @ bsp_basis

# making the part not fitted for proper sigma                                                  
for i in range(wsr_sigma.shape[0]):
    wsr_sigma[i, :-1018] = wsr_sigma[i, -1018]

# plotting 
r = summary['params']['dpy']['GVARS'].r

fig, ax = plt.subplots(1, 3, figsize=(10,5))

for i in range(3):
    idx = i + 2
    ax[i].plot(r, wsr_hybrid_full[idx] - wsr_dpt_full[idx], 'k', lw = 1)
    ax[i].fill_between(r, wsr_sigma[idx], -wsr_sigma[idx], color='red',
                       alpha=0.4)
    
    ax[i].grid(True)
    ax[i].set_xlim([0.88, 1])

    ax[i].set_xlabel('$r$ in $R_{\odot}$')
    ax[i].set_ylabel('$w_{%i}^{\mathrm{hybrid}}(r) - w_{%i}^{\mathrm{DPT}}(r)$ in $\mu$Hz'%(2*idx+1, 2*idx+1))

left  = 0.07  # the left side of the subplots of the figure
right = 0.98    # the right side of the subplots of the figure
bottom = 0.12   # the bottom of the subplots of the figure
top = 0.94      # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
# plt.tight_layout()

plt.savefig(f'dpy-hybrid-{PARGS.daynum}.pdf')


# plotting full profile
fig, ax = plt.subplots(3, 1, figsize=(4,9))

for i in range(3):
    idx = i + 2
    ax[i].plot(r, wsr_dpt_full[idx], 'k', lw=0.2, label='DPT')
    ax[i].plot(r, wsr_hybrid_full[idx], 'r', lw=0.2, label='Hybrid')
    ax[i].fill_between(r, wsr_dpt_full[idx] - abs(wsr_sigma[idx]), 
                       wsr_dpt_full[idx] + abs(wsr_sigma[idx]), color='blue',
                       alpha=0.2)
    ax[i].grid(True)
    # ax[i].set_ylabel('$w_{%i}(r) - w_{%i}^{\mathrm{ref}}(r)$ in $\mu$Hz'%(2*idx+1, 2*idx+1))
    ax[i].set_ylabel('$w_{%i}(r)$ in $\mu$Hz'%(2*idx+1))
    ax[i].legend()

ax[2].set_xlabel('$r$ in $R_{\odot}$')
plt.tight_layout()

plt.savefig(f'dpy-hybrid-full-{PARGS.daynum}.pdf')
