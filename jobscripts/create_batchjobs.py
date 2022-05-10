import subprocess
import argparse
import numpy as np
import os
import re

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python", shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
parallelpath = "/homes/hanasoge/parallel/bin/parallel"
jobname = f"sgk.init"
execpath = f"{package_dir}/jobscripts/batchjobs.sh"

gnup_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -joe
#PBS -o {jobname}.oe
#PBS -l select=1:ncpus=112:mem=700gb
#PBS -l walltime=12:00:00
#PBS -q clx
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
source activate jaxpyro
{pythonpath} {package_dir}/batch_run/initialize_batch_runs.py
{parallelpath} --jobs 12 < $PBS_O_WORKDIR/ipjobs_batch.sh
echo \"Finished at \"`date`
"""

slurm_str = f"""#!/bin/bash                                                                  
#SBATCH --job-name={jobname}                                                              
#SBATCH --output=out-{jobname}.log                                                            
#SBATCH --error=err-{jobname}.log
#SBATCH --nodes=1                                                                             
#SBATCH --ntasks-per-node=40                                                                  
#SBATCH --mem=180G                                                                            
#SBATCH --time=15:00:00                                                                       
echo \"Starting at \"`date`                                                                   
module purge                                                                                  
module load anaconda3                                                                         
conda activate jax-gpu                                                                        
echo \"Starting at \"`date`                                                                   
parallel --jobs 3 < {package_dir}/jobscripts/ipjobs_batch.sh                                
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_batch.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/gnup_batch.slurm", "w") as f:
    f.write(slurm_str)

bashdir = f"{package_dir}/jobscripts/bashbatch"
os.system(f"cd {bashdir}; ls *.sh | grep hybrid > hybridnames.txt")
os.system(f"cd {bashdir}; ls *.sh | grep -v hybrid > dpynames.txt")
with open(f"{bashdir}/dpynames.txt", "r") as f: dpynames = f.read().splitlines()
with open(f"{bashdir}/hybridnames.txt", "r") as f: hybridnames = f.read().splitlines()
batchnames = []
batchnames.extend(dpynames)
batchnames.extend(hybridnames)


with open(f"{package_dir}/jobscripts/ipjobs_batch.sh", "w") as f:
    for bname in batchnames:
        f.write(f"sh {bashdir}/{bname} &>logs/{bname}.oe\n")
