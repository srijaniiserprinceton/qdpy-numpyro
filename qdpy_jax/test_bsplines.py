from qdpy_jax import bsplines as bsp_jax
from scipy.interpolate import BSpline as bsp_scipy
import matplotlib.pyplot as plt
from scipy import interpolate
import jax
import numpy as np

_bsp_func_jax = jax.jit(bsp_jax.bspline1d)

if __name__ == "__main__":
    N = 1000
    x = np.linspace(0, 5, N)
    y = (x + 1.5*x*x)*np.sin(np.pi/2 + 2*np.pi*x/10.)

    k = 3
    t = np.array([0.1, 1.2, 2.3, 3.4, 4.5])
    t, c, __ = interpolate.splrep(x, y, k=k, t=t)
    bsp_func_scipy = bsp_scipy(t, c, k)

    x_new = np.linspace(0.2, 4.4, N)
    y_scipy = bsp_func_scipy(x_new)
    # y_jax = _bsp_func_jax(x_new, c, t, k)

    plt.figure()
    plt.plot(x, y, 'k', label='true')
    plt.plot(x_new, y_scipy, '--r', label='scipy')

    y_jax = bsp_jax.bspline1d(x_new, c[1:-k], t, k)
    plt.plot(x_new, y_jax, '-.g', label='jax')
