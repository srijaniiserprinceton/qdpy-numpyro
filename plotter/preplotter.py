import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import gen_wsr

def plot_extreme_wsr(r, r_spline, wsr_dpt, ctrl_arr_upex, ctrl_arr_loex,
                     knot_arr, rth_ind, spl_deg=3):
    wsr_upex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_upex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    wsr_loex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_loex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    plt.plot(r, wsr_loex[0, :], 'b', label='wsr_loex')
    plt.plot(r, wsr_upex[0, :], 'r', label='wsr_upex')

    plt.legend()

    plt.savefig('wsr_extreme.pdf')
