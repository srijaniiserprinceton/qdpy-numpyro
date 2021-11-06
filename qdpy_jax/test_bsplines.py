from qdpy_jax import bsplines as bsp_adams
from scipy.interpolate import BSpline as bsp_scipy
import matplotlib.pyplot as plt
from scipy import interpolate
import jax
import numpy as np

_bsp_func_jax = jax.jit(bsp_adams.bspline1d)

if __name__ == "__main__":
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

    y_scipy = bsp_func_scipy(x_new)

    plt.figure()
    # plt.plot(x, y, 'k', label='true')
    plt.plot(x_new, y_scipy, 'k', label='scipy')

    y_adams= bsp_adams.bspline1d(x_new, c, t, k)
    # y_adams = _bsp_func_jax(x_new, c, t, k)
    plt.plot(x_new, y_adams, '-.r', label='adams')
    plt.legend()
    plt.show()
