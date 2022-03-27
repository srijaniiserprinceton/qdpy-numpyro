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

samarth_dpy_dir = f'{scratch_dir}/carr_dpt_samarth'

# the list of days to extract
pt = pd.read_table(f'{scratch_dir}/input_files/daylist.txt', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

# getting the bsp_basis_clipped from one of my files
batchrun_dir = f'{scratch_dir}/batch_runs_hybrid'
daydir = f'{batchrun_dir}/hmi_72d_10000_18'
summary_fname = os.listdir(f'{daydir}/summaryfiles')[0]
summary = jf.load_obj_with_pklext(f'{daydir}/summaryfiles/{summary_fname}')
bsp_basis_clipped = summary['params']['dpy']['GVARS'].bsp_basis

# the set of files computed by Samarth
dayind_min, dayind_max = 0, 30

baddays = np.array([10432, 9352, 10072, 9712, 10288])

for dayind in range(dayind_min, dayind_max+1):       
    day = pt['MDI'][dayind]
    print(day)
    if(day in baddays): 
        print(f'Bad day: {day}; skipping.')
        continue
    daydir = f'{batchrun_dir}/hmi_72d_{day}_18'

    # saving the wsr from dpy
    carr_dpy_fit = np.load(f'{samarth_dpy_dir}/carr_dpt_{day}.npy')
    carr_dpy_fit = np.reshape(carr_dpy_fit, (3,-1), 'F')
    wsr_dpy_fit = carr_dpy_fit @ bsp_basis_clipped

    # saving the DPT and hybrid wsr profiles
    np.save(f'wsr_dpy_fit_{day}.npy', wsr_dpy_fit)

# saving the radial grid (which should be the same for all inverions)
np.save('r.npy', summary['params']['dpy']['GVARS'].r)

# saving GVARS.OM
np.save('GVARS_OM.npy', summary['params']['dpy']['GVARS'].OM)
