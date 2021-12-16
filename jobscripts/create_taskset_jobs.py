import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--numchains", help="Total number of chains",
                    type=int, default=2)
parser.add_argument("--maxiter", help="Maximum iteration number",
                    type=int, default=2)
parser.add_argument("--warmup", help="Number of warmup steps",
                    type=int, default=2)
ARGS = parser.parse_args()

numchains = ARGS.numchains
maxiter = ARGS.maxiter
warmup = ARGS.warmup


current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
jobname = f"sgk.qdpt.{numchains}"
execpath = f"{package_dir}/vorontsov_qdpy/run_reduced_problem_model.py"
logdir = f"{package_dir}/jobscripts/logs"

job_str = \
f"""#!/bin/bash
#PBS -N {jobname}
#PBS -o out-{jobname}.log
#PBS -e err-{jobname}.log
#PBS -l select=1:ncpus={numchains}:mem={2*numchains}gb
#PBS -l walltime=45:00:00
#PBS -q clx
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
cd ..
PYTHONPATH={pythonpath}
EXECPATH={execpath}
LOGDIR={logdir}
"""
for i in range(numchains):
    ipjobs_str = f"taskset -c {i:3d} $PYTHONPATH $EXECPATH "
    job_args = f"--warmup {warmup} --maxiter {maxiter} --chain_num {i:3d}"
    outfile = f" >$LOGDIR/qd.{i:03d}.out"
    errfile = f" 2>$LOGDIR/qd.{i:03d}.err"
    job_str = job_str + ipjobs_str + job_args + outfile + errfile + "\n"

job_str = job_str + "echo \"Finished at \"`date`"

with open(f"{package_dir}/jobscripts/gnup_taskset.pbs", "w") as f:
    f.write(job_str)
