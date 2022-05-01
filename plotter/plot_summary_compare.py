import os
import numpy as np
from datetime import date
from datetime import datetime
from qdpy import jax_functions as jf
from plotter import postplotter
import subprocess
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

dpy_dir = f"{scratch_dir}/dpy_jax"
qdpy_dir = f"{scratch_dir}/qdpy_jax"
#------------------------------------------------------------------------# 
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="output directory",
                    type=str, default=f"{scratch_dir}/summaryfiles")
PARGS = parser.parse_args()
#------------------------------------------------------------------------#

"""
dirnames = subprocess.check_output(["ls", f"{PARGS.outdir}"])
dirnames = dirnames.decode().split('\n')
numdir = len(dirnames)
for i in range(numdir):
    print(f"[ {i:2d} ] {dirnames[i]}")
idx = int(input(f"Select dataset to be plotted: "))
outdir = f"{PARGS.outdir}/{dirnames[idx]}/summaryfiles"
outdir = f"{PARGS.outdir}/{dirnames[idx]}/plots"
print(f"outdir = {outdir}")
"""
outdir = f"{PARGS.outdir}"
plotdir = f"{PARGS.outdir}"


def select_and_load():
    os.system(f"ls {outdir}/summary* > {outdir}/fnames.txt")
    with open(f"{outdir}/fnames.txt", "r") as f:
        fnames = f.read().splitlines()

    for i in range(len(fnames)):
        print(f"{i:^5d} | {fnames[i]}")

    select_modes = True
    summary_list = []
    count = 1

    while select_modes:
        selector = input(f"File [{count}] | Enter the index for filename " +
                         f"(enter x to exit) :")
        if selector == 'x':
            select_modes = False
            break
        summary = jf.load_obj(f"{fnames[int(selector)][:-4]}")
        summary_list.append(summary)
        count += 1
    return summary_list


def plot_from_summary(summlist):
    fig, ax = None, None
    colors = ['red', 'blue', 'magenta', 'black', 'orange']
    count = 0
    for summary in summlist:
        GVARS = summary['params']['dpy']['GVARS']

        c_arr_fit = summary['c_arr_fit']
        true_params_flat = summary['true_params_flat']
        cind_arr = summary['cind_arr']
        sind_arr = summary['sind_arr']

        suffix = f"{int(GVARS.knot_num)}s.{GVARS.eigtype}.{GVARS.tslen}d"
        c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                        sind_arr, cind_arr)
        fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full,
                                           c_arr_fit_full*0.0, f'summary-{suffix}')
        fig, ax = fit_plot.plot_fit_wsr(fig=fig, ax=ax, pcolor=colors[count])
        count += 1
    return fig


if __name__ == "__main__":
    summary_list = select_and_load()
    fig = plot_from_summary(summary_list)
    fig.savefig(f"{plotdir}/compare-dpt-qdpt.pdf")
    
