import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from dpy_jax import jax_functions as jf
from dpy_jax import globalvars as gvar_jax

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
parser.add_argument("--rth", help="rth",
                    type=float, default=0.9)
parser.add_argument("--knot_num", help="num knots beyond rth",
                    type=int, default=15)
parser.add_argument("--maxiter", help="max MCMC iterations",
                    type=int, default=100)
ARGS = parser.parse_args()

GVARS = gvar_jax.GlobalVars(n0=int(ARGS.n0),
                            lmin=int(ARGS.lmin),
                            lmax=int(ARGS.lmax),
                            rth=ARGS.rth,
                            knot_num=int(ARGS.knot_num),
                            load_from_file=0)

def plot_chains(wnum):
    key_prefix = f"c{wnum}_"
    key_list = [k1 for k1 in sample_keys if key_prefix in k1]
    print(key_list)
    num_plots = len(key_list)
    fig, axs = plt.subplots(nrows=num_plots, ncols=2, figsize=(5, 2*num_plots))
    axs = axs.reshape(num_plots, 2)
    ploti = 0
    for i in range(num_plots-1, -1, -1):
        # this_key = key_prefix + f"{i}"
        this_key = key_list[i]
        idx = int(this_key.split("_")[-1])
        cmin = limits['cmin'][this_key]
        cmax = limits['cmax'][this_key]
        axs[ploti, 0].plot(samples1[this_key])
        axs[ploti, 0].axhline(y=GVARS.ctrl_arr_dpt_clipped[int((wnum-1)//2), idx],
                              color='red', alpha=0.5)
        axs[ploti, 0].set_xlabel('Iteration number')
        axs[ploti, 0].set_ylabel(this_key)
        axs[ploti, 0].set_ylim([cmin, cmax])

        axs[ploti, 1].hist(samples1[this_key])
        axs[ploti, 1].set_ylabel('Count')
        # axs[ploti, 1].set_xlim([cmin, cmax])
        axs[ploti, 1].axvline(x=GVARS.ctrl_arr_dpt_clipped[int((wnum-1)//2), idx],
                              color='red', alpha=0.5)
        ploti += 1
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fname = f"output-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
    output_data = jf.load_obj(f"{dirnames[1]}/{fname}")

    samples1 = output_data['samples']
    limits = output_data['ctrl_limits']
    metadata = output_data['metadata']

    sample_keys = samples1.keys()
    for ic in np.array([3, 5], dtype=np.int32):
        fig = plot_chains(ic)
        fig.savefig(f"{dirnames[1]}/c{ic}.png")
        plt.close(fig)
