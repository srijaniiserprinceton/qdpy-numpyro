import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

from qdpy_jax import jax_functions as jf
from qdpy_jax import gen_wsr
from qdpy_jax import globalvars as jgvars

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"

with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()

parser = argparse.ArgumentParser()
parser.add_argument("--n0", help="radial order",
                    type=int, default=0)
parser.add_argument("--lmin", help="min angular degree",
                    type=int, default=200)
parser.add_argument("--lmax", help="max angular degree",
                    type=int, default=200)
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
ARGS = parser.parse_args()


def plot_wsr_extreme():
    # getting the dpt profile from clipped dpt
    ctrl_arr_dpt_recon = np.zeros_like(GVARS.ctrl_arr_dpt_full)
    ctrl_arr_dpt_recon[:, GVARS.knot_ind_th:] = ctrl_arr
    wsr_dpt_recon = gen_wsr.get_wsr_from_spline(GVARS.r, ctrl_arr_dpt_recon,
                                                GVARS.t_internal, GVARS.spl_deg)

    # getting the upex profile
    ctrl_arr_up_full = np.zeros_like(GVARS.ctrl_arr_dpt_full)
    ctrl_arr_up_full[:, GVARS.knot_ind_th:] = ctrl_arr_up
    wsr_up_recon = gen_wsr.get_wsr_from_spline(GVARS.r, ctrl_arr_up_full,
                                                GVARS.t_internal, GVARS.spl_deg)

    # getting the loex profile
    ctrl_arr_lo_full = np.zeros_like(GVARS.ctrl_arr_dpt_full)
    ctrl_arr_lo_full[:, GVARS.knot_ind_th:] = ctrl_arr_lo
    wsr_lo_recon = gen_wsr.get_wsr_from_spline(GVARS.r, ctrl_arr_lo_full,
                                                GVARS.t_internal, GVARS.spl_deg)
    # getting the loex profile
    # plotting 
    fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex = True)

    # the dpt reconstructed from spline
    ax[0].plot(GVARS.r, GVARS.wsr_fixed[0] + wsr_dpt_recon[0], 'k')
    ax[1].plot(GVARS.r, GVARS.wsr_fixed[1] + wsr_dpt_recon[1], 'k')
    ax[2].plot(GVARS.r, GVARS.wsr_fixed[2] + wsr_dpt_recon[2], 'k')

    ax[0].plot(GVARS.r, GVARS.wsr[0], '--r')
    ax[1].plot(GVARS.r, GVARS.wsr[1], '--r')
    ax[2].plot(GVARS.r, GVARS.wsr[2], '--r')

    # saving the profile being used in 1walk1iter_sparse.py forusing in qdpt.py
    wsr_pyro = GVARS.wsr_fixed + wsr_dpt_recon

    # shading the area where it is allowed to vary
    ax[0].fill_between(GVARS.r, GVARS.wsr_fixed[0] + wsr_up_recon[0],
                        GVARS.wsr_fixed[0] + wsr_lo_recon[0],
                        color='gray', alpha=0.5)

    ax[1].fill_between(GVARS.r, GVARS.wsr_fixed[1] + wsr_up_recon[1],
                        GVARS.wsr_fixed[1] + wsr_lo_recon[1],
                        color='gray', alpha=0.5)

    ax[2].fill_between(GVARS.r, GVARS.wsr_fixed[2] + wsr_up_recon[2],
                        GVARS.wsr_fixed[2] + wsr_lo_recon[2],
                        color='gray', alpha=0.5)

    fig.tight_layout()
    return fig



def build_ctrlarr_from_sample(samples, wnum):
    key_prefix = f"c{wnum}_"
    key_list = [k1 for k1 in sample_keys if key_prefix in k1]
    num_params = len(key_list)

    c_arr_mean = np.zeros(num_params+4)
    c_arr_up = np.zeros(num_params+4)
    c_arr_lo = np.zeros(num_params+4)

    for i in range(num_params):
        mean_val = samples[key_list[i]].mean()
        std_dev = samples[key_list[i]].std()
        c_arr_mean[i] = mean_val
        c_arr_up[i] = mean_val + std_dev
        c_arr_lo[i] = mean_val - std_dev

    return c_arr_mean, c_arr_up, c_arr_lo

if __name__ == "__main__":
    fname = f"output-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
    output_data = jf.load_obj(f"{dirnames[1]}/{fname}")

    samples1 = output_data['samples']
    limits = output_data['ctrl_limits']
    metadata = output_data['metadata']

    GVARS = jgvars.GlobalVars(n0=metadata['n0'],
                              lmin=metadata['lmin'],
                              lmax=metadata['lmax'],
                              rth=metadata['rth'],
                              knot_num=metadata['knot_num'])

    sample_keys = samples1.keys()
    ctrl_arr, ctrl_arr_up, ctrl_arr_lo = build_ctrlarr_from_sample(samples1, 1)

    for wnum in [3, 5]:
        c1, cup, clo = build_ctrlarr_from_sample(samples1, wnum)
        ctrl_arr = np.vstack((ctrl_arr, c1))
        ctrl_arr_up = np.vstack((ctrl_arr_up, cup))
        ctrl_arr_lo = np.vstack((ctrl_arr_lo, clo))

    fig = plot_wsr_extreme()
    fig.savefig(f"{dirnames[1]}/wsr-output.pdf")
    plt.close(fig)
