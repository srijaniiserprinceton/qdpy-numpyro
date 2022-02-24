from ritzLavelyPy import rlclass as RLC
from qdpy import globalvars as gvar_jax
import numpy as np
import argparse
import os
#------------------------- argument parser -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="dpy or qdpy",
                    type=str, default='dpy_jax')
parser.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
parser.add_argument("--tslen", help="72 or 360",
                    type=int, default=72)
parser.add_argument("--daynum", help="day from MDI epoch",
                    type=int, default=6328)
parser.add_argument("--numsplits", help="number of splitting coefficients",
                    type=int, default=18)
PARGS = parser.parse_args()
#-----------------------------------------------------------------------

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
outdir = f"{scratch_dir}/{PARGS.outdir}"
ipdir = f"{package_dir}/{PARGS.outdir}"

#-----------------------------------------------------------------------
def gen_RL_poly():
    ellmax = np.max(GVARS.ell0_arr)
    RL_poly = np.zeros((len(GVARS.ell0_arr), jmax+1, 2*ellmax+1), dtype=np.float64)

    for ell_i, ell in enumerate(GVARS.ell0_arr):
        RLP = RLC.ritzLavelyPoly(ell, jmax)
        RL_poly[ell_i, :, :2*ell+1] = RLP.Pjl

    return RL_poly
#-----------------------------------------------------------------------

if __name__ == '__main__':
    ARGS = np.loadtxt(f"{ipdir}/.n0-lmin-lmax.dat")
    GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                                lmin=int(ARGS[1]),
                                lmax=int(ARGS[2]),
                                rth=ARGS[3],
                                knot_num=int(ARGS[4]),
                                load_from_file=int(ARGS[5]),
                                relpath=outdir,
                                instrument=PARGS.instrument,
                                tslen=PARGS.tslen,
                                daynum=PARGS.daynum,
                                numsplits=PARGS.numsplits)

    jmax = GVARS.smax
    RL_poly = gen_RL_poly()
    print(f"Shape = {RL_poly.shape}")
    sfx = GVARS.filename_suffix
    np.save(f'{outdir}/RL_poly.{sfx}.npy', RL_poly)
