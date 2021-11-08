import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import gen_wsr

def plot_extreme_wsr(r, r_spline, OM, wsr_dpt, ctrl_arr_upex, ctrl_arr_loex,
                     knot_arr, rth_ind, spl_deg=3):

    wsr_upex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_upex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    wsr_loex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_loex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    fig, ax = plt.subplots(3,1,figsize=(15,10))

    s_arr = np.array([1,3,5])

    for i, s in enumerate(s_arr):
        ax[i].plot(r, wsr_loex[i, :]*OM*1e9, 'b')
        ax[i].plot(r, wsr_upex[i, :]*OM*1e9, 'r')
        ax[i].plot(r, wsr_dpt[i,:]*OM*1e9, '--k')

    plt.savefig('wsr_extreme.pdf')

    
