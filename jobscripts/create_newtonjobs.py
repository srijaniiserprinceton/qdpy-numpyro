import subprocess
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mumin", help="Min regularization",
                    type=float, default=1e-6)
parser.add_argument("--mumax", help="Max regularization",
                    type=float, default=1e3)
parser.add_argument("--s", help="s",
                    type=int, default=3)
ARGS = parser.parse_args()

muexp_min = np.log10(ARGS.mumax)
muexp_max = np.log10(ARGS.mumin)
muexp_list = np.linspace(muexp_min, muexp_max, 500)
mu_list = 10**muexp_list
print(f"mu = {mu_list}")

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
execpath = f"{package_dir}/dpy_jax/run_reduced_problem_lcurve.py"
rundir = f"/tier2/tifr/samarth/qdpy_iofiles/batch_runs_dpy/hmi_72d_6328_18"
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
{pythonpath} /homes/hanasoge/samarth/qdpy-numpyro/batch_run/batch_precompute.py --rundir {rundir} --s {sval}
{pythonpath} {execpath} --store_hess 1 --instrument {instr} --mu 1.0 --batch_run 1 --batch_rundir {rundir} --s {sval}
/homes/hanasoge/parallel/bin/parallel --jobs 80 < $PBS_O_WORKDIR/ipjobs_lcurve_s{sval}.sh
echo \"Finished at \"`date`
"""

with open(f"{package_dir}/jobscripts/gnup_lcurve_s{sval}.pbs", "w") as f:
    f.write(gnup_str)

with open(f"{package_dir}/jobscripts/ipjobs_lcurve_s{sval}.sh", "w") as f:
    for idx, muval in enumerate(mu_list):
        ipjobs_str = f"{pythonpath} {execpath} "
        job_args = (f"--read_hess 1 --instrument {instr} " +
                    f"--mu {muval} --batch_run 1 --batch_rundir {rundir} --plot 1 " +
                    f"--s {sval}")
        oefile = f" &> {package_dir}/jobscripts/logs/qdrls.s{sval}.{idx:03d}.oe"
        f.write(ipjobs_str + job_args + oefile + " \n")
