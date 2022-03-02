import os
import re
import numpy as np
import argparse
import fnmatch

#-----------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--batch_run", help="flag to indicate its a batch run",
                    type=int, default=0)
parser.add_argument("--batch_rundir", help="local directory for batch run",
                    type=str, default=".")
PARGS = parser.parse_args()
#-----------------------------------------------------------------------#

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

# reading the instrument from the rundir                                                      
local_rundir = re.split('[/]+', ARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]
#-----------------------------------------------------------------------# 

def compute_misfit(arr1, arr2):
    return np.sqrt(sum(abs(arr1 - arr2)**2))

if(not PARGS.batch_run):
    outdir = f"{scratch_dir}/dpy_jax"
else:
    outdir = f"{PARGS.batch_rundir}"

run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"

mu_limits = [1e-15, 1e-3]
print(f"Running python scripts")

os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} \
            --mu {mu_limits[0]} --batch_run 1 --batch_rundir {PARGS.rundir} \
            >{PARGS.rundir}/temp.out 2>{PARGS.rundir}/emp.err")
os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} \
            --mu {mu_limit[1]} --batch_run 1 --batch_rundir {PARGS.rundir} \
            >{PARGS.rundir}/temp.out 2>{PARGS.rundir}/emp.err")

# val0 corresponds to iterative solution
fname = fnmatch.filter(os.listdir(PARGS.batch_rundir), 'carr_iterative_*.npy')[0]
val0 = np.load(f'{PARGS.batch_rundir}/{fname}')


val1 = np.load(f'{outdir}/carr_fit_{mu_limits[0]:.5e}.npy')
val2 = np.load(f'{outdir}/carr_fit_{mu_limits[1]:.5e}.npy')
mf1 = compute_misfit(val0, val1)
mf2 = compute_misfit(val0, val2)
print(f"muval1 = {mu_limits[0]:.5e}; misfit = {mf1:.5e}")
print(f"muval1 = {mu_limits[1]:.5e}; misfit = {mf2:.5e}")

if mf1 > mf2:
    leftmu = mu_limits[0]
    rightmu = mu_limits[1]
else:
    leftval = mu_limits[1]
    rightval = mu_limits[0]

maxiter = 20

for i in range(maxiter):
    log_muval = 0.5*(np.log10(leftmu) + np.log10(rightmu))
    muval = 10**log_muval
    os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} \
                --mu {muval} --batch_run 1 --batch_rundir {PARGS.rundir} \
                >{PARGS.rundir}/temp.out 2>{PARGS.rundir}/emp.err")

    muval1 = 10**(0.5*(np.log10(leftmu) + log_muval))
    muval2 = 10**(0.5*(np.log10(rightmu) + log_muval))
    os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} \
                --mu {muval1} --batch_run 1 --batch_rundir {PARGS.rundir} \
                >{PARGS.rundir}/temp.out 2>{PARGS.rundir}/emp.err")
    os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} \
                --mu {muval2} --batch_run 1 --batch_rundir {PARGS.rundir} \
                >{PARGS.rundir}/temp.out 2>{PARGS.rundir}/emp.err")
    
    val1 = np.load(f'{outdir}/carr_fit_{muval1:.5e}.npy')
    val2 = np.load(f'{outdir}/carr_fit_{muval2:.5e}.npy')
    mf1 = compute_misfit(val0, val1)
    mf2 = compute_misfit(val0, val2)
    print(f"muval1 = {muval1:.5e}; misfit = {mf1:.5e}")

    if mf1 < mf2:
        leftmu = muval1 * 1.0
        rightmu = muval * 1.0
    else:
        leftmu = muval * 1.0
        rightmu = muval2 * 1.0
    
