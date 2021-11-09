import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"

with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()



def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


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
    samples1 = load_obj(f"{dirnames[1]}/samples")
    sample_keys = samples1.keys()
    fig = plot_chains(1)
    fig.show()
