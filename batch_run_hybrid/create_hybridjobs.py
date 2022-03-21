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
execpath = f"{package_dir}/hybrid_jax/run_reduced_hybrid_iterative.py"
dpy_iterpath = f"{package_dir}/dpy_jax/run_reduced_problem_iterative.py"

batchnames = [filename for
              filename in os.listdir(f"{scratch_dir}/batch_runs_hybrid")
              if (os.path.isdir(f"{scratch_dir}/batch_runs_hybrid/{filename}")
                  and filename[0]!='.')]
print(batchnames)

count = 0
num_batches = len(batchnames)
num_jobs = 1
for i in range(num_jobs):
    job_args = []
    jobdpy_args = []
    oe_files = []
    oedpy_files = []
    bn_start = batchnames[num_jobs*i]
    for j in range(num_batches//num_jobs):
        bname = batchnames[num_jobs*j + i]
        print(f"Creating job for {bname}")
        batch_hybrid_dir = f"{scratch_dir}/batch_runs_hybrid/{bname}"
        mu_batchdir = f"{scratch_dir}/batch_runs_dpy/{bname}"
        batch_rundir = f"{batch_hybrid_dir}/dpy_full_hess"
        instr = bname.split('_')[0]
        job_str = f"{pythonpath} {execpath} "
        jobdpy_str = f"{pythonpath} {dpy_iterpath} "
        _jobdpy_args = (f"--mu 1.0 --batch_run 1 --store_hess 1 " +
                        f"--instrument {instr} --batch_rundir {batch_rundir}")
        _job_args = (f"--mu 1.0 --instrument {instr} --mu_batchdir {mu_batchdir} " +
                     f"--rundir {batch_hybrid_dir}")
        _oedpy_file = f"{package_dir}/jobscripts/logs/dpy-{bname}.oe"
        _oe_file = f"{package_dir}/jobscripts/logs/hybrid-{bname}.oe"
        job_args.append(_job_args)
        jobdpy_args.append(_jobdpy_args)
        oe_files.append(_oe_file)
        oedpy_files.append(_oedpy_file)

    jobname = f"hybrid.{i}"
    gnup_str = \
        f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus=112:mem=700gb
#PBS -l walltime=60:00:00
#PBS -q clx
echo \"Starting at \"`date`
echo \"First dataset = {bn_start}\"
echo \"Last dataset = {bname}\"
"""
    slurm_str = \
    f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=150G
#SBATCH --time=05:00:00
echo \"Starting at \"`date`
module purge
module load anaconda3
conda activate jax-gpu    
"""
    for j in range(num_batches//num_jobs):
        # gnup_str += f"{jobdpy_str} {jobdpy_args[j]} &>{oedpy_files[j]}\n"
        gnup_str += f"{job_str} {job_args[j]} &>{oe_files[j]}\n"
        # slurm_str += f"{jobdpy_str} {jobdpy_args[j]} &>{oedpy_files[j]}\n"
        slurm_str += f"{job_str} {job_args[j]} &>{oe_files[j]}\n"
    
    gnup_str = gnup_str + f"""echo \"Finished at \"`date`"""
    slurm_str = slurm_str + f"""echo \"Finished at \"`date`"""

    with open(f"{package_dir}/jobscripts/gnup_hybrid_{i}.pbs", "w") as f:
        f.write(gnup_str)

    with open(f"{package_dir}/jobscripts/gnup_hybrid_{i}.slurm", "w") as f:
        f.write(slurm_str)
