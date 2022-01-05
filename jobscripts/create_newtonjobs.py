import subprocess
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mumin", help="Min regularization",
                    type=float, default=2)
parser.add_argument("--mumax", help="Max regularization",
                    type=float, default=2)
ARGS = parser.parse_args()

muexp_min = np.log10(ARGS.mumax)
muexp_max = np.log10(ARGS.mumin)
muexp_list = np.linspace(muexp_min, muexp_max, 32)
mu_list = 10**muexp_list

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
jobname = f"dpt.RLS"
execpath = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"

gnup_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=06:00:00
#PBS -q small
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
cd ..
parallel --jobs 32 < $PBS_O_WORKDIR/ipjobs_dpt_rls.sh
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_dpt_rls.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/ipjobs_dpt_rls.sh", "w") as f:
    for idx, muval in enumerate(mu_list):
        ipjobs_str = f"{pythonpath} {execpath} "
        job_args = f"--mu {muval}"
        outfile = f" >{package_dir}/jobscripts/qdrls.{idx:03d}.out"
        errfile = f" 2>{package_dir}/jobscripts/qdrls.{idx:03d}.err"
        f.write(ipjobs_str + job_args + outfile + errfile + "\n")
