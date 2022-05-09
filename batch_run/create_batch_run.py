import os
import re
import sys
import numpy as np
import subprocess
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
smax_global = int(dirnames[3])

parser = argparse.ArgumentParser()
parser.add_argument("--jobtype", default="pbs",
                    type=str, help="pbs/slurm")
PARGS = parser.parse_args()

batch_dir = f"{scratch_dir}/batch_runs_dpy"
bashscr_dir = f"{package_dir}/jobscripts/bashbatch"
#----------------- getting full pythonpath -----------------------
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------
batchnames = [filename for filename in os.listdir(batch_dir) if 
              (os.path.isdir(f"{batch_dir}/{filename}") and filename[0]!='.')]

smax_local = smax_global * 1
lcurve_slist = [7]
run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"
run_lcurve_py = f"{package_dir}/jobscripts/create_lcurve_jobs.py"

for bname in batchnames:
    instr = re.split('[_]+', bname, flags=re.IGNORECASE)[0]
    with open(f"{bashscr_dir}/{bname}.sh", "w") as f:
        f.write("#!/bin/sh\n")
        
        for s in range(1, smax_local+1, 2):
            f.write(f"{pythonpath} {current_dir}/batch_precompute.py " +
                    f"--rundir {batch_dir}/{bname} --s {s}\n")
            f.write(f"{pythonpath} {current_dir}/mu_gss.py --instr {instr} "
                    f"--rundir {batch_dir}/{bname} --s {s}\n")
            f.write(f"cp {batch_dir}/{bname}/dhess*.npy " +
                    f"{batch_dir}/{bname}/s{s}_dhess.npy\n")
            f.write(f"cp {batch_dir}/{bname}/mhess*.npy " +
                    f"{batch_dir}/{bname}/s{s}_mhess.npy\n")
            f.write(f"cp {batch_dir}/{bname}/carr_fit_1.00000e+00.npy " +
                    f"{batch_dir}/{bname}/s{s}_carr.npy\n")
            f.write(f"cp {batch_dir}/{bname}/carr_fit_scaled_1.00000e+00.npy " +
                    f"{batch_dir}/{bname}/s{s}_carr_scaled.npy\n")
            f.write(f"{pythonpath} {current_dir}/check_mugss.py " +
                    f"--fname {batch_dir}/{bname}/s{s}_carr_scaled.npy\n")
        
       # all s fitting with the optimal mu
        mu = 1.0
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py "
                f"--rundir {batch_dir}/{bname} --s 0\n")
        f.write(f"{pythonpath} {run_newton_py} "
                f"--mu {mu} --s 0 --instrument {instr} " +
                f"--batch_run 1 --batch_rundir {batch_dir}/{bname} " +
                f"--mu_batchdir {batch_dir}/{bname}\n")
        f.write(f"cp {batch_dir}/{bname}/plots/fit_{mu:.5e}_wsr.pdf " +
                f"{batch_dir}/{bname}/plots/fit_wsr_sall_optimal_mu.pdf\n")

        """
        for s in range(1, smax_local+1, 2):
            for lcs in lcurve_slist:
                if s == lcs:
                    f.write(f"{pythonpath} {current_dir}/batch_precompute.py " +
                            f"--rundir {batch_dir}/{bname} --s {s}\n")
                    f.write(f"{pythonpath} {run_lcurve_py} --jobtype {PARGS.jobtype} "
                            f"--rundir {batch_dir}/{bname} --s {s} --instr {instr}\n")
        """
    os.system(f"chmod u+x {bashscr_dir}/{bname}.sh")
