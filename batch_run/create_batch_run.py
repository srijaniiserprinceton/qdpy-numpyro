import os
import re
import sys
import numpy as np
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
batch_dir = f"{scratch_dir}/batch_runs_dpy"
bashscr_dir = f"{package_dir}/jobscripts/bashbatch"

#----------------- getting full pythonpath -----------------------
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------
batchnames = [filename for filename in os.listdir(batch_dir) if 
              (os.path.isdir(f"{batch_dir}/{filename}") and filename[0]!='.')]

mu_start = np.array([1.e-4, 1.e-5, 1.e-4])

# previously used for s=1, 3, 5
# f.write(f"{pythonpath} {current_dir}/batch_iterative_inversion.py --rundir {batch_dir}/{bname} --s 1 --mu {mu_start[0]}\n")
# f.write(f"cp {batch_dir}/{bname}/plots/fit_wsr.pdf {batch_dir}/{bname}/plots/fit_wsr_s1_iter.pdf\n")

run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"
run_iterative_py = f"{package_dir}/dpy_jax/run_reduced_problem_iterative.py"

for bname in batchnames:
    instr = re.split('[_]+', bname, flags=re.IGNORECASE)[0]
    with open(f"{bashscr_dir}/{bname}.sh", "w") as f:
        f.write("#!/bin/sh\n")
        # s = 1
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py " +
                f"--rundir {batch_dir}/{bname} --s 1\n")
        f.write(f"{pythonpath} {current_dir}/mu_gss.py "
                f"--rundir {batch_dir}/{bname} --s 1\n")

        # s = 3
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py " +
                f"--rundir {batch_dir}/{bname} --s 3\n")
        f.write(f"{pythonpath} {current_dir}/mu_gss.py " +
                f"--rundir {batch_dir}/{bname} --s 3\n")
        
        # s = 5
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py " +
                f"--rundir {batch_dir}/{bname} --s 5\n")
        f.write(f"{pythonpath} {current_dir}/mu_gss.py " +
                f"--rundir {batch_dir}/{bname} --s 5\n")
        
        # all s fitting with the optimal mu
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py "
                f"--rundir {batch_dir}/{bname} --s 0\n")
        f.write(f"{pythonpath} {run_newton_py} "
                f"--mu 1.0 --store_hess 1 --s 0 --instrument {instr} " +
                f"--batch_run 1 --batch_rundir {batch_dir}/{bname} " +
                f"--mu_batchdir {batch_dir}/{bname}\n")
        f.write(f"cp {batch_dir}/{bname}/plots/fit_wsr.pdf {batch_dir}/{bname}/plots/fit_wsr_sall_optimal_mu.pdf\n")
        f.write(f"{pythonpath} {run_iterative_py} " +
                f"--mu 1.0 --s 0 --instrument {instr} " +
                f"--batch_run 1 --batch_rundir {batch_dir}/{bname}\n") #--read_hess 1 
        f.write(f"cp {batch_dir}/{bname}/plots/fit_wsr.pdf {batch_dir}/{bname}/plots/fit_wsr_sall_iter.pdf\n")
        
    os.system(f"chmod u+x {bashscr_dir}/{bname}.sh")
