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
            int(RP[8]), int(RP[9]), int(RP[10]))

#----------------------READING THE RUN DIRECTORY--------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--rundir", help="local directory for one batch run",
                    type=str)
parser.add_argument("--s", help="the s case to run",
                    type=int)
ARGS = parser.parse_args()

#-----------------------------QDPY DIRECTORY------------------------------# 
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)

# reading the instrument from the rundir
local_rundir = re.split('[/]+', ARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]

#-------------------------READING THE RUNPARAMS---------------------------#
if(ARGS.s == 1):
    RPARAMS = np.loadtxt(f"{ARGS.rundir}/.params_smin_1_smax_1.dat")

    nmin, nmax, lmin, lmax, smin, smax, knotnum, rth, tslen, daynum, numsplits =\
                                                                params2vars(RPARAMS)

    #-------------------------------MODE LISTER-------------------------------#
    mode_lister_py = f"{package_dir}/qdpy/mode_lister.py"
    
    outdir_wrt_scratchout = f"batch_runs_dpy/{local_rundir}"
    mlist_out = f"{ARGS.rundir}/.mlist.out"
    mlist_err = f"{ARGS.rundir}/.mlist.err"
    
    os.system(f'python {mode_lister_py} --nmin {nmin} --nmax {nmax}\
                --lmin {lmin} --lmax {lmax} --instrument {instr} --tslen {tslen}\
                --daynum {daynum} --numsplits {numsplits} --outdir {outdir_wrt_scratchout}\
                --exclude_qdpy 0 >{mlist_out} 2>{mlist_err}')
