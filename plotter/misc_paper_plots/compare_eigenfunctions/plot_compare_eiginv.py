import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from qdpy import jax_functions as jf

# directory in which the output is stored
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

inveig_dir = f'{scratch_dir}/inversion-eigfuncs'

# loading the summaryfiles
summary_jesper = jf.load_obj(f'{inveig_dir}/summary.dpt-24.03.2022-19.26-15s.Jesper-0')
summary_antia = jf.load_obj(f'{inveig_dir}/summary.dpt-24.03.2022-21.34-15s.Antia-0')
summary_modelS = jf.load_obj(f'{inveig_dir}/summary.dpt-24.03.2022-22.12-15s.modelS-0')
summary_gyre = jf.load_obj(f'{inveig_dir}/summary.dpt-24.03.2022-23.23-15s.gyre-0')

# loading the final fitted .npy files to compare with the summary file carr_fit
carr_fit_jesper = np.load(f'{inveig_dir}/carr_fit_Jesper.npy')
carr_fit_antia = np.load(f'{inveig_dir}/carr_fit_Antia.npy')
carr_fit_modelS = np.load(f'{inveig_dir}/carr_fit_modelS.npy')
carr_fit_gyre = np.load(f'{inveig_dir}/carr_fit_gyre.npy')

# loading the final fitted carr_fit from the summary files and the true_params
carr_fit_jesper_summary = summary_jesper['c_arr_fit']
carr_fit_antia_summary = summary_antia['c_arr_fit']
carr_fit_modelS_summary = summary_modelS['c_arr_fit']
carr_fit_gyre_summary = summary_gyre['c_arr_fit']

true_params_jesper_summary = summary_jesper['true_params_flat']
true_params_antia_summary = summary_antia['true_params_flat']
true_params_modelS_summary = summary_modelS['true_params_flat']
true_params_gyre_summary = summary_gyre['true_params_flat']

# testing
np.testing.assert_array_almost_equal(carr_fit_jesper_summary,
                                     carr_fit_jesper/true_params_jesper_summary)

np.testing.assert_array_almost_equal(carr_fit_antia_summary,
                                     carr_fit_antia/true_params_antia_summary)

np.testing.assert_array_almost_equal(carr_fit_modelS_summary,
                                     carr_fit_modelS/true_params_modelS_summary)

np.testing.assert_array_almost_equal(carr_fit_gyre_summary,
                                     carr_fit_gyre/true_params_gyre_summary)


# making the wsr profiles
bsp_basis_jesper = summary_jesper['params']['dpy']['GVARS'].bsp_basis_full
bsp_basis_antia = summary_antia['params']['dpy']['GVARS'].bsp_basis_full
bsp_basis_modelS = summary_modelS['params']['dpy']['GVARS'].bsp_basis_full
bsp_basis_gyre = summary_gyre['params']['dpy']['GVARS'].bsp_basis_full

sind_arr = np.array([0,1,2])
cind_arr = np.arange(len(carr_fit_jesper)//3)

carr_fit_jesper_full = jf.c4fit_2_c4plot(summary_jesper['params']['dpy']['GVARS'],
                                         carr_fit_jesper, sind_arr, cind_arr)
carr_fit_antia_full = jf.c4fit_2_c4plot(summary_antia['params']['dpy']['GVARS'],
                                         carr_fit_antia, sind_arr, cind_arr)
carr_fit_modelS_full = jf.c4fit_2_c4plot(summary_modelS['params']['dpy']['GVARS'],
                                         carr_fit_modelS, sind_arr, cind_arr)
carr_fit_gyre_full = jf.c4fit_2_c4plot(summary_gyre['params']['dpy']['GVARS'],
                                         carr_fit_gyre, sind_arr, cind_arr)


wsr_jesper = carr_fit_jesper_full @ bsp_basis_jesper
wsr_antia = carr_fit_antia_full @ bsp_basis_antia
wsr_modelS = carr_fit_modelS_full @ bsp_basis_modelS
wsr_gyre = carr_fit_gyre_full @ bsp_basis_gyre

# plotting 
r_jesper = summary_jesper['params']['dpy']['GVARS'].r
r_antia = summary_antia['params']['dpy']['GVARS'].r
r_modelS = summary_modelS['params']['dpy']['GVARS'].r
r_gyre = summary_gyre['params']['dpy']['GVARS'].r

fig, ax = plt.subplots(3, 1, figsize=(4,9))

for i in range(3):
    ax[i].plot(r_antia, wsr_antia[i] - wsr_jesper[i], 'k', lw = 1, label='EFN I')
    ax[i].plot(r_modelS, wsr_modelS[i] - wsr_jesper[i], '--k', lw = 1, label='EFN II')
    ax[i].plot(r_gyre, wsr_gyre[i] - wsr_jesper[i], '-.k', lw = 1, label='EFN III')
    
    ax[i].grid(True)
    ax[i].set_xlim([0.88, 1])
    ax[i].set_ylabel('$w_{%i}(r) - w_{%i}^{\mathrm{ref}}(r)$ in $\mu$Hz'%(2*i+1, 2*i+1))

ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
             ncol=3, fancybox=False, shadow=False)

ax[2].set_xlabel('$r$ in $R_{\odot}$')

plt.tight_layout()

plt.savefig('compare_eiginv.pdf')
