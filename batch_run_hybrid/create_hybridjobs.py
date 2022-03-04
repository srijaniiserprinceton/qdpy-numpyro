import subprocess
import numpy as np
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
execpath = f"{package_dir}/hybrid_jax/run_reduced_problem_hybrid_batch.py"

batchnames = [filename for filename in os.listdir(f"{scratch_dir}/batch_runs_hybrid") if 
              (os.path.isdir(f"{scratch_dir}/batch_runs_hybrid/{filename}") and filename[0]!='.')]
print(batchnames)

for bname in batchnames:
    print(f"Creating job for {bname}")
    batch_hybrid_dir = f"{scratch_dir}/batch_runs_hybrid/{bname}"
    mu_batchdir = f"{scratch_dir}/batch_runs/{bname}"
    instr = bname.split('_')[0]
    job_str = f"{pythonpath} {execpath} "
    job_args = (f"--mu 1.0 --instrument {instr} --mu_batchdir {mu_batchdir} " +
                f"--rundir {batch_hybrid_dir}")
    jobname = f"hybrid-{bname}"
    gnup_str = \
    f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=32:mem=70gb
#PBS -l walltime=06:00:00
#PBS -q small
echo \"Starting at \"`date`
    
{job_str} {job_args}
echo \"Finished at \"`date`
"""

    with open(f"{package_dir}/jobscripts/gnup_hybrid_{bname}.pbs", "w") as f:
        f.write(gnup_str)
