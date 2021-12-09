from ritzLavelyPy import rlclass as RLC
from qdpy_jax import globalvars as gvar_jax
import numpy as np


def gen_RL_poly():
    ellmax = np.max(GVARS.ell0_arr)
    RL_poly = np.zeros((len(GVARS.ell0_arr), jmax+1, 2*ellmax+1), dtype=np.float64)

    for ell_i, ell in enumerate(GVARS.ell0_arr):
        RLP = RLC.ritzLavelyPoly(ell, jmax)
        RL_poly[ell_i, :, :2*ell+1] = RLP.Pjl

    return RL_poly


if __name__ == '__main__':
    ARGS = np.loadtxt(".n0-lmin-lmax.dat")
    GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                                lmin=int(ARGS[1]),
                                lmax=int(ARGS[2]),
                                rth=ARGS[3],
                                knot_num=int(ARGS[4]),
                                load_from_file=int(ARGS[5]))

    jmax = GVARS.smax
    RL_poly = gen_RL_poly()
    print(f"Shape = {RL_poly.shape}")
    np.save('RL_poly.npy', RL_poly)
