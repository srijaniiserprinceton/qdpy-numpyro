import os
import re
import sys
import fnmatch
import argparse
import numpy as np

def params2vars(RP):
    '''Function to convert the read params to passable arguments.
    '''
    return (int(RP[0]), int(RP[1]), int(RP[2]), int(RP[3]),
            int(RP[4]), int(RP[5]), int(RP[6]), float(RP[7]),
            int(RP[8]), int(RP[9]), int(RP[10]), int(RP[11]), int(RP[12]))

#----------------------READING THE RUN DIRECTORY--------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--rundir", help="local directory for one batch run",
                    type=str)
parser.add_argument("--full_qdpy_dpy", help="to run full or qdpy or dpy",
                    type=str)
ARGS = parser.parse_args()

#-----------------------------QDPY DIRECTORY------------------------------# 
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()

smax_global = int(dirnames[3])
smax_local = smax_global

# reading the instrument from the rundir
local_rundir = re.split('[/]+', ARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]

# defining the directory to contain the full dpy_jax fun for hessian
if(ARGS.full_qdpy_dpy == 'full'):
    rundir = f"{ARGS.rundir}/dpy_full_hess"
    outdir_wrt_scratchout = f"batch_runs_hybrid/{local_rundir}/dpy_full_hess"
if(ARGS.full_qdpy_dpy == 'qdpy'):
    rundir = f"{ARGS.rundir}/qdpy_files"
    outdir_wrt_scratchout = f"batch_runs_hybrid/{local_rundir}/qdpy_files"
if(ARGS.full_qdpy_dpy == 'dpy'):
    rundir = f"{ARGS.rundir}/dpy_files"
    outdir_wrt_scratchout = f"batch_runs_hybrid/{local_rundir}/dpy_files"

#-------------------------READING THE RUNPARAMS---------------------------#
# RPARAMS = np.loadtxt(f"{rundir}/.params_smin_1_smax_{smax_global}.dat")
RPARAMS = np.loadtxt(f"{rundir}/.params_smin_1_smax_{smax_local}.dat")

nmin, nmax, lmin, lmax, smin, smax, knotnum, rth, tslen, daynum, numsplits, exclude_qdpy, __=\
                                                                        params2vars(RPARAMS)

#-------------------------------MODE LISTER-------------------------------#
mode_lister_py = f"{package_dir}/qdpy/mode_lister.py"

mlist_out = f"{rundir}/.mlist.out"
mlist_err = f"{rundir}/.mlist.err"

os.system(f'python {mode_lister_py} --nmin {nmin} --nmax {nmax} --batch_run 1 \
            --lmin {lmin} --lmax {lmax} --instrument {instr} --tslen {tslen} \
            --daynum {daynum} --numsplits {numsplits} --outdir {outdir_wrt_scratchout} \
            --exclude_qdpy {exclude_qdpy} --smax_global {smax_global}\
            >{mlist_out} 2>{mlist_err}')

#-------------------------------GENERATE---------------------------------#
if (not ARGS.full_qdpy_dpy == 'qdpy'):
    generate_py = f"{package_dir}/dpy_jax/generate_synthetic_eigvals.py"
    
    # copying the sparsee_precompute_acoeff.py to the local run directory
    os.system(f"cp {package_dir}/dpy_jax/sparse_precompute_acoeff.py \
                {rundir}/sparse_precompute_acoeff_batch.py")
    
    os.system(f"python {generate_py} --load_mults 1 \
               --knot_num {knotnum} --rth {rth} --instrument {instr} \
               --tslen {tslen} --daynum {daynum} --numsplits {numsplits} \
               --batch_run 1 --batch_rundir {rundir} --smax_global {smax_global}")

#-------------------------SAVE REDUCED PROBLEM-----------------------------#
if (ARGS.full_qdpy_dpy == 'qdpy'):
    save_reduced_py = f"{package_dir}/qdpy_jax/save_reduced_problem.py"
    # copying the sparsee_precompute_acoeff.py to the local run directory                     
    os.system(f"cp {package_dir}/qdpy_jax/sparse_precompute.py \
                {rundir}/sparse_precompute_batch.py")
    # copying the .config to the local run directory to be used by sparse_precompute        
    os.system(f"cp {package_dir}/.config {ARGS.rundir}/.")
    
    os.system(f"python {save_reduced_py} --instrument {instr} --load_mults 1\
               --rth {rth} --knot_num {knotnum} --tslen {tslen} --daynum {daynum}\
               --numsplits {numsplits} --batch_run 1 --batch_rundir {rundir}\
               --smax_global {smax_global} --smax {smax_local}")

else:
    save_reduced_py = f"{package_dir}/dpy_jax/save_reduced_problem.py"
    
    os.system(f"python {save_reduced_py} --instrument {instr} --smin {smin} \
                --smax {smax_local} --batch_run 1 --batch_rundir {rundir}")


#------------------------RITZLAVELY POLYNOMIALS---------------------------#
rlpoly_py = f"{package_dir}/qdpy/precompute_ritzlavely.py"
rl_out = f"{rundir}/.rl.out"
rl_err = f"{rundir}/.rl.err"

os.system(f"python {rlpoly_py} --outdir {rundir} --instrument {instr} \
            --batch_run 1 >{rl_out} 2>{rl_err}")


