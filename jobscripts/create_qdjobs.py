import subprocess
import os
import argparse

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

gnup_str = \
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
parallel --jobs {numchains} < $PBS_O_WORKDIR/ipjobs_v11.sh
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_v11.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/ipjobs_v11.sh", "w") as f:
    for i in range(numchains):
        ipjobs_str = f"{pythonpath} {execpath} "
        job_args = f"--warmup {warmup} --maxiter {maxiter} --chain_num {i}"
        outfile = f" >{package_dir}/jobscripts/qd.{i:03d}.out"
        errfile = f" 2>{package_dir}/jobscripts/qd.{i:03d}.err"
        f.write(ipjobs_str + job_args + outfile + errfile + "\n")
