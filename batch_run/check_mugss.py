import os
import numpy as np
import argparse
import fnmatch
import subprocess
from qdpy import globalvars as gvar_jax
from scipy.integrate import simps


#-----------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="local directory for batch run",
                    type=str, default=".")
PARGS = parser.parse_args()
#-----------------------------------------------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
smax_global = int(dirnames[-2])
INSTR = dirnames[-1]

_pythonpath = subprocess.check_output("which python", shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------------#
fname = PARGS.fname
npyname = fname.split('/')[-1]
sval = int(npyname.split('_')[0][1])
batchdir = ''
for k in fname.split('/')[:-1]:
    batchdir += k + '/'


def check_ones(carr, threshold=0.25):
    num_data = len(carr)
    num_near1 = (abs(carr - 1) >= threshold).sum()
    near1_fraction = num_near1 / num_data
    return near1_fraction

carr = np.load(PARGS.fname)
n1f = check_ones(carr)
np.save(f"{batchdir}/s{sval}_carr_closenessfraction.npy", n1f)
