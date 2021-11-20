import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from qdpy_jax import jax_functions as jf

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


def plot_chains(wnum):
    key_prefix = f"c{wnum}_"
    key_list = [k1 for k1 in sample_keys if key_prefix in k1]
    num_plots = len(key_list)
    fig, axs = plt.subplots(nrows=num_plots, ncols=2, figsize=(5, 2*num_plots))
    for i in range(num_plots):
        cmin = limits['cmin'][key_list[i]]
        cmax = limits['cmax'][key_list[i]]
        axs[i, 0].plot(samples1[key_list[i]])
        axs[i, 0].set_xlabel('Iteration number')
        axs[i, 0].set_ylabel(key_list[i])
        axs[i, 0].set_ylim([cmin, cmax])

        axs[i, 1].hist(samples1[key_list[i]])
        axs[i, 1].set_ylabel('Count')
        axs[i, 1].set_xlim([cmin, cmax])
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fname = f"output-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
    output_data = jf.load_obj(f"{dirnames[1]}/{fname}")

    samples1 = output_data['samples']
    limits = output_data['ctrl_limits']
    metadata = output_data['metadata']

    sample_keys = samples1.keys()
    for ic in np.array([1, 3, 5], dtype=np.int32):
        fig = plot_chains(ic)
        fig.savefig(f"{dirnames[1]}/c{ic}.pdf")
        plt.close(fig)
