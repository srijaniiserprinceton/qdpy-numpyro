import subprocess
import argparse
import numpy as np
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python", shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
jobname = f"sgk.hyb-init"
execpath = f"{package_dir}/jobscripts/batchjobs.sh"

gnup_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=112:mem=700gb
#PBS -l walltime=12:00:00
#PBS -q clx
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
{pythonpath} {package_dir}/batch_run/initialize_batch_runs.py
parallel --jobs 12 < $PBS_O_WORKDIR/ipjobs_batch.sh
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_batch.pbs", "w") as f:
    f.write(gnup_str)

bashdir = f"{package_dir}/jobscripts/bashbatch"
with open(f"{package_dir}/jobscripts/ipjobs_batch.sh", "w") as f:
    batchnames = [filename for filename in os.listdir(bashdir) if 
                  (filename[-3:] == '.sh')]
    for bname in batchnames:
        f.write(f"sh {bashdir}/{bname} >logs/{bname}.out 2>logs/{bname}.err\n")
