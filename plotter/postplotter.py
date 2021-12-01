import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

import jax.numpy as jnp
from dpy_jax import jax_functions as jf
from dpy_jax import globalvars as jgvars
from qdpy_jax import gen_wsr
from dpy_jax import sparse_precompute as precompute
from dpy_jax import build_hypermatrix_sparse as build_hm_sparse

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
parser.add_argument("--load_mults", help="load multiplets from file",
                    type=int, default=0)
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

    c_arr_mean = GVARS.ctrl_arr_dpt_clipped[int((wnum-1)//2), :]
    c_arr_up = GVARS.ctrl_arr_dpt_clipped[int((wnum-1)//2), :]
    c_arr_lo = GVARS.ctrl_arr_dpt_clipped[int((wnum-1)//2), :]

    for i in range(num_params):
        # this_key = key_prefix + f"{i}"
        this_key = key_list[i]
        idx = int(this_key.split("_")[-1])
        mean_val = samples[this_key][discard:].mean()
        std_dev = samples[this_key][discard:].std()
        c_arr_mean[idx] = mean_val
        c_arr_up[idx] = mean_val + std_dev
        c_arr_lo[idx] = mean_val - std_dev

    return c_arr_mean, c_arr_up, c_arr_lo

def plot_eigs(eigsample, eigtrue, eigsigma,
              eigsample_up, eigsample_lo):
    eigs = eigsample[:401]
    eigs_up = eigsample_up[:401]
    eigs_lo = eigsample_lo[:401]
    eigt = eigtrue[:401]
    eigsig = eigsigma[:401]
    fig = plt.figure()
    plt.plot(eigs - eigt, 'k')
    plt.fill_between(np.arange(401), -eigsig, eigsig, color='gray')
    plt.fill_between(np.arange(401), eigs_lo-eigs, eigs-eigs_up, color='red',
                     alpha=0.5)
    return fig

if __name__ == "__main__":
    fname = f"output-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
    output_data = jf.load_obj(f"{dirnames[1]}/{fname}")
    discard = 50

    samples1 = output_data['samples']
    sample_keys = samples1.keys()
    limits = output_data['ctrl_limits']
    metadata = output_data['metadata']

    noc_hypmat_all_sparse, fixed_hypmat_all_sparse, omega0_arr =\
        precompute.build_hypmat_all_cenmults()

    GVARS = jgvars.GlobalVars(n0=metadata['n0'],
                              lmin=metadata['lmin'],
                              lmax=metadata['lmax'],
                              rth=metadata['rth'],
                              knot_num=metadata['knot_num'],
                              load_from_file=ARGS.load_mults)
    GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
    eigvals_true = jnp.asarray(GVARS_TR.eigvals_true)
    eigvals_sigma = jnp.asarray(GVARS_TR.eigvals_sigma)

    # ctrl_arr, ctrl_arr_up, ctrl_arr_lo = build_ctrlarr_from_sample(samples1, 1)

    ctrl_arr = GVARS.ctrl_arr_dpt_clipped[0, :]
    ctrl_arr_up = GVARS.ctrl_arr_dpt_clipped[0, :]*1.03
    ctrl_arr_lo = GVARS.ctrl_arr_dpt_clipped[0, :]*0.97

    for wnum in GVARS.s_arr[1:]:
        c1, cup, clo = build_ctrlarr_from_sample(samples1, wnum)
        ctrl_arr = np.vstack((ctrl_arr, c1))
        ctrl_arr_up = np.vstack((ctrl_arr_up, cup))
        ctrl_arr_lo = np.vstack((ctrl_arr_lo, clo))

    fig = plot_wsr_extreme()
    fig.savefig(f"{dirnames[1]}/wsr-output.pdf")
    plt.close(fig)

    len_s = len(GVARS.s_arr)
    diag_evals = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                  fixed_hypmat_all_sparse,
                                                  ctrl_arr,
                                                  GVARS.nc,
                                                  len_s)
    diag_evals = diag_evals.todense()/2./omega0_arr*GVARS.OM*1e6

    diag_evals_up = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                     fixed_hypmat_all_sparse,
                                                     ctrl_arr_up,
                                                     GVARS.nc,
                                                     len_s)
    diag_evals_up = diag_evals_up.todense()/2./omega0_arr*GVARS.OM*1e6

    diag_evals_lo = build_hm_sparse.build_hypmat_w_c(noc_hypmat_all_sparse,
                                                     fixed_hypmat_all_sparse,
                                                     ctrl_arr_lo,
                                                     GVARS.nc,
                                                     len_s)
    diag_evals_lo = diag_evals_lo.todense()/2./omega0_arr*GVARS.OM*1e6

    fig = plot_eigs(diag_evals, eigvals_true, eigvals_sigma,
                    diag_evals_up,
                    diag_evals_lo)
    fig.savefig(f"{dirnames[1]}/eigs.pdf")
    plt.close(fig)
