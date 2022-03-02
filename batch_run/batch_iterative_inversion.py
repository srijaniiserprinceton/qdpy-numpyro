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
parser.add_argument("--mu", help="regularization",
                    type=float, default=0.)
ARGS = parser.parse_args()

#-----------------------------QDPY DIRECTORY------------------------------#                   
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)

# reading the instrument from the rundir                                                      
local_rundir = re.split('[/]+', ARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]

#--------------------------NEWTON RUN TO STORE HESS-----------------------#
run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"

os.system(f"python {run_newton_py} --mu {ARGS.mu} --store_hess 1 \
            --instrument {instr} --batch_run 1 --batch_rundir {ARGS.rundir}")

#------------------------------ITERATIVE RUN------------------------------#
run_iterative_py = f"{package_dir}/dpy_jax/run_reduced_problem_iterative.py"

os.system(f"python {run_iterative_py} --mu {ARGS.mu} --read_hess 1 \
            --instrument {instr} --batch_run 1 --batch_rundir {ARGS.rundir}")
