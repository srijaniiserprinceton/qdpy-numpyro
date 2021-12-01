import numpy as np
from scipy import integrate
from scipy.interpolate import splrep

from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import sparse_precompute as sp_precomp
GVARS = gvar_jax.GlobalVars()

def test_bsp_integral():
    r = GVARS.r
    wsr = GVARS.wsr

    # integrating without spline route. Shape (3,)
    int_1 = integrate.simps(wsr, r, axis=1)

    # obtaining the spline coefficients
    len_s = wsr.shape[0]

    # to find the length of ctrl vector
    __, c, __ = splrep(r, wsr[0])

    c_arr = np.zeros((len_s, len(c)))

    for i in range(len_s):
        t, c, k = splrep(r, wsr[i])
        c_arr[i] = c

    # finding the integral of basis elements from our custom function
    bsp_params = (c_arr.shape[1], t, k)
    bsp_integrated = sp_precomp.integrate_bsp_basis_elements(r, bsp_params)

    # finally weighing the integrals with the corresponding control points
    int_2 = c_arr @ bsp_integrated

    np.testing.assert_array_almost_equal(int_1, int_2)

if __name__=='__main__':
    test_bsp_integral()
