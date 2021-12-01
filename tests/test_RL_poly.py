from dpy_jax import ritzlavely as rl
from dpy_jax import globalvars as gvar_jax
import numpy as np

GVARS = gvar_jax.GlobalVars()

def test_Pjl():
    RL_poly_200_5_qdPy = np.load('rlp_200_5.npy')
    
    make_RL_poly = rl.ritzLavelyPoly(GVARS)
    RL_poly = make_RL_poly.RL_poly[0]
    
    np.testing.assert_array_equal(RL_poly, RL_poly_200_5_qdPy)

if __name__ == '__main__':
    test_Pjl()
