from qdpy_jax import bsplines as bsp_adams
from scipy.interpolate import BSpline as bsp_scipy
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import splev, splrep
import jax
import numpy as np
from qdpy_jax import globalvars as gvar_jax

from jax.config import config
config.update('jax_enable_x64', True)

bspline_1d_ = jax.jit(bsp_adams.bspline1d, static_argnums=(3,))

def test_samarth():
    N = 1000
    k = 2
    knotstart = 0.1
    knotend = 4.5
    startarr = np.ones(k+1)*knotstart
    endarr = np.ones(k+1)*knotend
    t = np.append(startarr, np.linspace(0.1, 4.5, 20))
    t = np.append(t, endarr)
    c = np.random.rand(len(t)-k-1)
    x_new = np.linspace(t.min()+0.005, t.max()-0.005, N)
    bsp_func_scipy = bsp_scipy(t, c, k)

    print(t.shape, c.shape, k)

    # reconstructing the B-splines
    y_scipy = bsp_func_scipy(x_new)
    y_adams= bsp_adams.bspline1d(x_new, c, t, k)

    # testing
    np.testing.assert_array_almost_equal(y_scipy, y_adams, decimal=10)


def test_srijan(idx):
    '''
    # getting the GVARS for the radius and the directories
    GVARS = gvar_jax.GlobalVars()
    r = GVARS.r
    wsr = GVARS.wsr[0]

    print(GVARS.rmax_ind, GVARS.rmin_ind)
    '''
    
    r = np.loadtxt('r.dat')[1:-1]
    wsr =  np.load('wsr-spline.npy')[idx] 
    print(r.shape, wsr.shape)
    
    # parameterizing in terms of cubic splines
    spl = splrep(r, wsr)

    # getting the knot vector, control points, degree
    t, c, k = spl

    # reconstructing the B-splines using scipy function
    y_scipy = bsp_scipy(t, c, k)(r)
    
    # adjusting the zero-padding in c from splrep
    c = c[:-(k+1)]

    # testing essential shapes criterion
    np.testing.assert_equal(len(t), len(c) + k + 1)

    # reconstructing the B-spline using Ryan P. Adams code
    # y_adams = bsp_adams.bspline1d(r, c, t, k)
    y_adams = bspline_1d_(r, c, t, k)

    return r, wsr, y_scipy, y_adams


if __name__ == "__main__":
    test_samarth()

    widx = 0
    r, wsr, y_sp, y_ad = test_srijan(widx)
    tolerance = 1e-16
    print(f"for tol={tolerance}: is y_scipy close to y_adams? " +
          f"{np.isclose(y_sp, y_ad, rtol=tolerance).all()}")
    # np.testing.assert_array_equal(y_sp, y_ad)
    np.testing.assert_array_almost_equal(y_sp, y_ad, decimal=13)

    plt.figure()
    plt.plot(r, wsr)
    plt.plot(r, y_ad, '--r')
    plt.show()
