import subprocess
import argparse
import numpy as np
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]


parser = argparse.ArgumentParser()
parser.add_argument("--mumin", help="Min regularization",
                    type=float, default=1e-6)
parser.add_argument("--mumax", help="Max regularization",
                    type=float, default=1e3)
parser.add_argument("--s", help="s",
                    type=int, default=3)
parser.add_argument("--rundir", type=str,
                    default="/tier2/tifr/samarth/qdpy_iofiles/batch_runs_dpy/hmi_72d_6328_18")
parser.add_argument("--jobtype", default="pbs",
                    type=str, help="pbs/slurm")
ARGS = parser.parse_args()

muexp_min = np.log10(ARGS.mumax)
muexp_max = np.log10(ARGS.mumin)
muexp_list = np.linspace(muexp_min, muexp_max, 500)
mu_list = 10**muexp_list
print(f"mu = {mu_list}")

execpath = f"{package_dir}/dpy_jax/run_reduced_problem_lcurve.py"
rundir = ARGS.rundir
sval = ARGS.s
instr = "hmi"

jobname = f"s{sval}.Lcurve"

gnup_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=112:mem=700gb
#PBS -l walltime=06:00:00
#PBS -q clx
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
source activate jaxpyro
{pythonpath} {package_dir}/batch_run/batch_precompute.py --rundir {rundir} --s {sval}
{pythonpath} {execpath} --store_hess 1 --instrument {instr} --mu 1.0 --batch_run 1 --batch_rundir {rundir} --s {sval}
/homes/hanasoge/parallel/bin/parallel --jobs 80 < $PBS_O_WORKDIR/ipjobs_lcurve_s{sval}.sh
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
{pythonpath} {package_dir}/batch_run/batch_precompute.py --rundir {rundir} --s {sval}
{pythonpath} {execpath} --store_hess 1 --instrument {instr} --mu 1.0 --batch_run 1 --batch_rundir {rundir} --s {sval}
parallel --jobs 80 < {pacakge_dir}/jobscripts/ipjobs_lcurve_s{sval}.sh
echo \"Finished at \"`date`
"""


with open(f"{package_dir}/jobscripts/gnup_lcurve_s{sval}.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/gnup_lcurve_s{sval}.slurm", "w") as f:
    f.write(slurm_str)

with open(f"{package_dir}/jobscripts/ipjobs_lcurve_s{sval}.sh", "w") as f:
    for idx, muval in enumerate(mu_list):
        ipjobs_str = f"{pythonpath} {execpath} "
        job_args = (f"--read_hess 1 --instrument {instr} " +
                    f"--mu {muval} --batch_run 1 --batch_rundir {rundir} --plot 1 " +
                    f"--s {sval}")
        oefile = f" &> {package_dir}/jobscripts/logs/qdrls.s{sval}.{idx:03d}.oe"
        f.write(ipjobs_str + job_args + oefile + " \n")

if ARGS.jobtype == "pbs":
    os.system(f"cd {pacakge_dir}/jobscripts; qsub gnup_lcurve_s{sval}.pbs")
elif ARGS.jobtype == "slurm":
    os.system(f"cd {pacakge_dir}/jobscripts; squeue gnup_lcurve_s{sval}.slurm")
else:
    print(f"invalid jobtype {ARGS.jobtype}. Can take only pbs or slurm")
