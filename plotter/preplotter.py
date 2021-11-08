import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import gen_wsr

def plot_extreme_wsr(r, r_spline, OM, wsr_dpt,
                     ctrl_arr_upex, ctrl_arr_dpt, ctrl_arr_loex,
                     knot_arr, rth_ind, spl_deg=3):

    wsr_upex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_upex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    wsr_dpt2 = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_dpt, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    wsr_loex = gen_wsr.get_wsr_from_spline(r_spline, wsr_dpt,
                                           ctrl_arr_loex, knot_arr,
                                           rth_ind, spl_deg=spl_deg)

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax = ax.flatten()
    s_arr = np.array([1, 3, 5])

    for i, s in enumerate(s_arr):
        ax[i].plot(r, wsr_loex[i, :]*OM*1e9, 'b', label='loex')
        ax[i].plot(r, wsr_upex[i, :]*OM*1e9, 'r', label='upex')
        ax[i].plot(r, wsr_dpt2[i, :]*OM*1e9, '--k', label='dpt-spline')
        ax[i].plot(r, wsr_dpt[i, :]*OM*1e9, 'g', alpha=0.7, label='dpt')
        ax[i].legend()
        ax[i].set_title(f'w{2*i+1}')
    ax[-1].semilogy(r, get_percent_error(wsr_dpt[0,:], wsr_dpt2[0, :]), 'r', label='w1')
    ax[-1].semilogy(r, get_percent_error(wsr_dpt[1,:], wsr_dpt2[1, :]), 'g', label='w3')
    ax[-1].semilogy(r, get_percent_error(wsr_dpt[2,:], wsr_dpt2[2, :]), 'b', label='w5')
    ax[-1].set_title('$w_{dpt} - w^{spline}_{dpt}$')
    ax[-1].legend()
    fig.tight_layout()

    plt.savefig('wsr_extreme.pdf')

def get_percent_error(a1, a2):
    errperc = np.zeros_like(a1)
    diff = a1 - a2
    mask0 = a1==0
    errperc[~mask0] = abs(diff)[~mask0]*100/abs(a1)[~mask0]
    return errperc

    
