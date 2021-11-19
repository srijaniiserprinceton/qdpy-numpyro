import os
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
parser.add_argument("--cplot", help="= (1, 3, 5)?",
                    type=int, default=1)
ARGS = parser.parse_args()


def plot_chains(wnum):
    key_prefix = f"c{wnum}_"
    key_list = [k1 for k1 in sample_keys if key_prefix in k1]
    num_plots = len(key_list)
    fig, axs = plt.subplots(nrows=num_plots, ncols=2, figsize=(5, 2*num_plots))
    for i in range(num_plots):
        axs[i, 0].plot(samples1[key_list[i]])
        axs[i, 0].set_xlabel('Iteration number')
        axs[i, 0].set_ylabel(key_list[i])

        axs[i, 1].hist(samples1[key_list[i]])
        axs[i, 1].set_ylabel('Count')
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fname = f"samples-{ARGS.n0}-{ARGS.lmin}-{ARGS.lmax}-{ARGS.maxiter}"
    samples1 = jf.load_obj(f"{dirnames[1]}/{fname}")
    sample_keys = samples1.keys()
    fig = plot_chains(ARGS.cplot)
    fig.savefig(f"{dirnames[0]}/c{ARGS.cplot}.pdf")
    fig.show()
