import os
import sys
import numpy as np
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
batch_dir = f"{scratch_dir}/batch_runs_hybrid"
batch_dir_mu = f"{scratch_dir}/batch_runs_dpy"
bashscr_dir = f"{package_dir}/jobscripts/bashbatch"

#----------------- getting full pythonpath -----------------------
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------
batchnames = [filename for filename in os.listdir(batch_dir) if 
              (os.path.isdir(f"{batch_dir}/{filename}") and filename[0]!='.')]


run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"
precompute_py = f"{current_dir}/batch_precompute.py"
some_mu = 1.0

for bname in batchnames:
    instr = bname.split('_')[0]
    with open(f"{bashscr_dir}/{bname}-hybrid.sh", "w") as f:
        f.write(f"{pythonpath} {precompute_py} " +
                f"--rundir {batch_dir}/{bname} " +
                f"--full_qdpy_dpy full\n")
        f.write(f"{pythonpath} {precompute_py} " +
                f"--rundir {batch_dir}/{bname} " +
                f"--full_qdpy_dpy qdpy\n")
        f.write(f"{pythonpath} {precompute_py} " +
                f"--rundir {batch_dir}/{bname} " +
                f"--full_qdpy_dpy dpy\n")

        # copying optimal mu files from dpy batch directory
        f.write(f"cp {batch_dir_mu}/{bname}/muval*.npy " +
                f"{batch_dir}/{bname}/dpy_full_hess}/")
        
        f.write(f"{pythonpath} {run_newton_py} --batch_run 1 " +
                f"--batch_rundir {batch_dir}/{bname}/dpy_full_hess --mu {some_mu} " +
                f"--store_hess 1 --instrument {instr}\n")
