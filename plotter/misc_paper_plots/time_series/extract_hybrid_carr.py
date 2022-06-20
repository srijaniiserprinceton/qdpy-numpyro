import os
import numpy as np
import pandas as pd
from qdpy import jax_functions as jf

# directory in which the output is stored                                                     
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
INSTR = dirnames[4]
INSTR = "hmi"
batchrun_dir = f'{scratch_dir}/batch_runs_hybrid'
orgfiles_dir = f"{scratch_dir}/{INSTR}-run1/organized-files"

# the list of days to extract
pt = pd.read_table(f'{package_dir}/preprocess/daylist.{INSTR}', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

dayind_min, dayind_max = 0, 40
# baddays = np.array([10432, 9352, 10072, 9712, 10288])
baddays = []

for dayind in range(dayind_min, dayind_max+1):       
    day = pt['MDI'][dayind]
    if (day in baddays):  print(f'Bad day: {day}; skipping.')
    
    try:
        summary = jf.load_obj(f'{orgfiles_dir}/hybrid-summary/summary_{day}')
        summary_dpy = jf.load_obj(f'{orgfiles_dir}/dpy-summary/summary_{day}')
    except FileNotFoundError:
        print(f"Not found: {day}")
        continue

#    try:
#        carr_dpy_fit = np.load(f'{daydir}/dpy_full_hess/carr_fit_1.00000e+00.npy')
#    except FileNotFoundError:
#        continue

    # params to build the full wsr for plotting
    true_params_flat = summary['true_params_flat']
    carr_dpy_fit = summary_dpy['c_arr_fit'] * true_params_flat
    sind_arr, cind_arr = summary['sind_arr'], summary['cind_arr']
    bsp_basis_full = summary['params']['dpy']['GVARS'].bsp_basis_full

    # saving the wsr from dpy
    carr_dpt_fit_full = jf.c4fit_2_c4plot(summary['params']['dpy']['GVARS'],
                                          carr_dpy_fit, sind_arr, cind_arr)
    wsr_dpy_fit = carr_dpt_fit_full @ bsp_basis_full
    jf.save_npy(f'{orgfiles_dir}/plot_files/wsr_dpy_fit_{day}.npy', wsr_dpy_fit)

    # saving the wsr from hybrid
    carr_hybrid_fit = summary['c_arr_fit'] * true_params_flat
    for i in range(5):
        print(f"=========== s = {2*i+1}; day = {day} ==============")
        print(f"{summary['c_arr_fit'][i::5]}")
    carr_hybrid_fit_full = jf.c4fit_2_c4plot(summary['params']['dpy']['GVARS'],
                                             carr_hybrid_fit, sind_arr, cind_arr)
    wsr_hybrid_fit = carr_hybrid_fit_full @ bsp_basis_full

    # saving the DPT and hybrid wsr profiles
    jf.save_npy(f'{orgfiles_dir}/plot_files/wsr_hybrid_fit_{day}.npy', wsr_hybrid_fit)

# saving the radial grid (which should be the same for all inverions)
jf.save_npy(f'{orgfiles_dir}/plot_files/r.npy', summary['params']['dpy']['GVARS'].r)

# saving GVARS.OM
jf.save_npy(f'{orgfiles_dir}/plot_files/GVARS_OM.npy', summary['params']['dpy']['GVARS'].OM)
