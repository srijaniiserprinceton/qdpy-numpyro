import os
import re
import numpy as np
import argparse
import fnmatch
from qdpy import globalvars as gvar_jax
from scipy.integrate import simps


#-----------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--rundir", help="local directory for batch run",
                    type=str, default=".")
parser.add_argument("--s", help="s", type=int, default=1)
PARGS = parser.parse_args()
#-----------------------------------------------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

# reading the instrument from the rundir                                                      
local_rundir = re.split('[/]+', PARGS.rundir, flags=re.IGNORECASE)[-1]
instr = re.split('[_]+', local_rundir, flags=re.IGNORECASE)[0]

run_newton_py = f"{package_dir}/dpy_jax/run_reduced_problem_newton.py"
tempout = f"{PARGS.rundir}/temp.out"
temperr = f"{PARGS.rundir}/temp.err"
outdir = f"{PARGS.rundir}"
#-----------------------------------------------------------------------# 
GVARS = gvar_jax.GlobalVars(rth=0.9, knot_num=15, relpath=PARGS.rundir)


os.system(f"python {run_newton_py} --store_hess 1 --instrument {instr} " +
      f"--mu {1.0} --batch_run 1 --batch_rundir {PARGS.rundir} " +
      f">{tempout} 2>{temperr}")

def compute_misfit(arr1, arr2):
    return np.sqrt(sum(abs(arr1 - arr2)**2))


def compute_misfit_wsr(arr1, arr2):
    if PARGS.s == 1: sind = 0
    if PARGS.s == 3: sind = 1
    if PARGS.s == 5: sind = 2
    carr1 = GVARS.ctrl_arr_dpt_full * 1.0
    carr2 = GVARS.ctrl_arr_dpt_full * 1.0
    carr1[sind, GVARS.knot_ind_th:] = arr1
    carr2[sind, GVARS.knot_ind_th:] = arr2
    wsr1 = (carr1 @ GVARS.bsp_basis_full)[sind]
    wsr2 = (carr2 @ GVARS.bsp_basis_full)[sind]
    absdiff2 = abs(wsr1 - wsr2)**2
    return np.sqrt(simps(absdiff2, x=GVARS.r))



def f(mu1):
    os.system(f"python {run_newton_py} --read_hess 1 --instrument {instr} " +
    # os.system(f"python {run_newton_py} --instrument {instr} " +
              f"--mu {mu1} --batch_run 1 --batch_rundir {PARGS.rundir} " +
              f">{tempout} 2>{temperr}")
    
    val1 = np.load(f'{outdir}/carr_fit_{mu1:.5e}.npy')
    mf = compute_misfit_wsr(val0, val1)
    print(f" mu = {mu1:.5e}, misfit = {mf:.5e}")
    return mf


invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

def gssrec(f, a, b, tol=1.0, h=None, hp=None, c=None, d=None, fc=None, fd=None):
    """ Golden-section search, recursive.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gssrec(f, a, b, tol)
    >>> print (c, d)
    1.9999959837979107 2.0000050911830893
    """

    (a, b) = (min(a, b), max(a, b))
    hp = (10**b - 10**a)/(10**a)*100.
    if hp <= tol: return (a, b)
    
    if h is None: h = b - a
    if c is None: c = a + invphi2 * h
    if d is None: d = a + invphi * h
    if fc is None: fc = f(10**c)
    if fd is None: fd = f(10**d)
    if fc < fd:
        return gssrec(f, a, d, tol, h * invphi, c=None, fc=None, d=c, fd=fc)
    else:
        return gssrec(f, c, b, tol, h * invphi, c=d, fc=fd, d=None, fd=None)

#-----------------------------------------------------------------------# 
# val0 corresponds to iterative solution
# fname = fnmatch.filter(os.listdir(PARGS.rundir), 'carr_iterative_*.npy')[0]
fname = fnmatch.filter(os.listdir(PARGS.rundir), 'true_params_flat*.npy')[0]
val0 = np.load(f'{PARGS.rundir}/{fname}')

mu_limits = [1e-15, 1e-3]
muvals = gssrec(f, np.log10(mu_limits[0]), np.log10(mu_limits[1]))
np.save(f"{outdir}/muval.s{PARGS.s}.npy", 10**muvals[0])