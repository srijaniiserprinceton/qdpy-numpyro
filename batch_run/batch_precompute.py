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
            int(RP[8]), int(RP[9]), int(RP[10]), int(RP[12]))

#----------------------READING THE RUN DIRECTORY--------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--rundir", help="local directory for one batch run",
                    type=str)
parser.add_argument("--s", help="the s case to run; default is 0 which is all",
                    type=int, default=0)
ARGS = parser.parse_args()

#-----------------------------QDPY DIRECTORY------------------------------# 
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()

# reading the instrument from the rundir
local_rundir = re.split('[/]+', ARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]

SMAX_GLOBAL = int(dirnames[3])

if(ARGS.s == 1):
    #-------------------------READING THE RUNPARAMS---------------------------#
    RPARAMS = np.loadtxt(f"{ARGS.rundir}/.params_smin_1_smax_1.dat")

    nmin, nmax, lmin, lmax, smin, smax, knotnum, rth, tslen, daynum, numsplits, smax_global =\
                                                                params2vars(RPARAMS)

    #-------------------------------MODE LISTER-------------------------------#
    mode_lister_py = f"{package_dir}/qdpy/mode_lister.py"
    
    outdir_wrt_scratchout = f"batch_runs_dpy/{local_rundir}"
    mlist_out = f"{ARGS.rundir}/.mlist.out"
    mlist_err = f"{ARGS.rundir}/.mlist.err"
    
    os.system(f'python {mode_lister_py} --nmin {nmin} --nmax {nmax} --batch_run 1 \
                --lmin {lmin} --lmax {lmax} --instrument {instr} --tslen {tslen} \
                --daynum {daynum} --numsplits {numsplits} --outdir {outdir_wrt_scratchout} \
                --exclude_qdpy 0 --smax_global {smax_global} >{mlist_out} 2>{mlist_err}')

    #-------------------------------GENERATE---------------------------------#
    generate_py = f"{package_dir}/dpy_jax/generate_synthetic_eigvals.py"

    # copying the sparsee_precompute_acoeff.py to the local run directory
    os.system(f"cp {package_dir}/dpy_jax/sparse_precompute_acoeff.py \
                {ARGS.rundir}/sparse_precompute_acoeff_batch.py")


    os.system(f"python {generate_py} --load_mults 1 \
                --knot_num {knotnum} --rth {rth} --instrument {instr} \
                --tslen {tslen} --daynum {daynum} --numsplits {numsplits} \
                --batch_run 1 --batch_rundir {ARGS.rundir} --smax_global {smax_global}")
    
    #------------------------RITZLAVELY POLYNOMIALS---------------------------#
    rlpoly_py = f"{package_dir}/qdpy/precompute_ritzlavely.py"
    rl_out = f"{ARGS.rundir}/.rl.out"
    rl_err = f"{ARGS.rundir}/.rl.err"
    
    os.system(f"python {rlpoly_py} --outdir {ARGS.rundir} --instrument {instr} \
                --batch_run 1 >{rl_out} 2>{rl_err}")
    
    #-------------------------SAVE REDUCED PROBLEM-----------------------------#
    save_reduced_py = f"{package_dir}/dpy_jax/save_reduced_problem.py"
    
    os.system(f"python {save_reduced_py} --instrument {instr} --smin {smin} \
                --smax {smax} --batch_run 1 --batch_rundir {ARGS.rundir}")


# to generate the precomptued files for the entire inversion (all s)
elif(ARGS.s == 0):
    smax = 5
    #-------------------------READING THE RUNPARAMS---------------------------#               
    # RPARAMS = np.loadtxt(f"{ARGS.rundir}/.params_smin_1_smax_{SMAX_GLOBAL}.dat")
    RPARAMS = np.loadtxt(f"{ARGS.rundir}/.params_smin_1_smax_{smax}.dat")
    nmin, nmax, lmin, lmax, smin, smax, knotnum, rth, tslen, daynum, numsplits, smax_global =\
                                                                params2vars(RPARAMS)

    #-------------------------SAVE REDUCED PROBLEM-----------------------------#              
    save_reduced_py = f"{package_dir}/dpy_jax/save_reduced_problem.py"
    os.system(f"python {save_reduced_py} --instrument {instr} --smin {smin} \
                --smax {smax} --batch_run 1 --batch_rundir {ARGS.rundir}")


else:
    #-------------------------READING THE RUNPARAMS---------------------------#
    RPARAMS = np.loadtxt(f"{ARGS.rundir}/.params_smin_{ARGS.s}_smax_{ARGS.s}.dat")

    nmin, nmax, lmin, lmax, smin, smax, knotnum, rth, tslen, daynum, numsplits, smax_global =\
                                                                params2vars(RPARAMS)
    
    #-------------------------SAVE REDUCED PROBLEM-----------------------------#             
    save_reduced_py = f"{package_dir}/dpy_jax/save_reduced_problem.py"
    
    os.system(f"python {save_reduced_py} --instrument {instr} --smin {smin} \
                --smax {smax} --batch_run 1 --batch_rundir {ARGS.rundir}")
