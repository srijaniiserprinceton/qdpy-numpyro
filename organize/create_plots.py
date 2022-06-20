import pandas as pd
import numpy as np
import os
import subprocess
import sys

#-----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
INSTR = dirnames[4]
dpy_batchdir = f"{scratch_dir}/batch_runs_dpy"
hybrid_batchdir = f"{scratch_dir}/batch_runs_hybrid"

_pythonpath = subprocess.check_output("which python", shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
parallelpath = "/homes/hanasoge/parallel/bin/parallel"
#-----------------------------------------------------------------------


def copy_files(srcname, destname):
    try:
        os.system(f"cp {srcname} {destname}")
    except FileNotFoundError:
        return None
    print(f"Writing {destname}")
    return None



pt = pd.read_table(f'../preprocess/daylist.{INSTR}', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

for i in range(77):
    daynum = int(pt['MDI'][i])
    compute_str = (f"{pythonpath} " +
                f"{package_dir}/dpy_jax/run_reduced_problem_newton.py " +
                f"--mu 1.0 --s 0 --instrument {INSTR} --batch_run 1 " +
                f"--batch_rundir {dpy_batchdir}/{INSTR}_72d_{daynum}_36 " +
                f"--mu_batchdir {dpy_batchdir}/{INSTR}_72d_{daynum}_36 " +
                f"--plot 1")
    print(compute_str)


fitplot_dir = f"{scratch_dir}/dpy_kneemu"
if not (os.path.isdir(fitplot_dir)):
    os.mkdir(fitplot_dir)


for i in range(77):
    daynum = int(pt['MDI'][i])
    copy_str = (f"cp {dpy_batchdir}/{INSTR}_72d_{daynum}_36/plots/fit_1.00000e+00_wsr.pdf " +
               f"{scratch_dir}/dpy_kneemu/fit_{daynum}.pdf")
    print(copy_str)

