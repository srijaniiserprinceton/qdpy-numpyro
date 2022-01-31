import os
import numpy as np
from datetime import date
from datetime import datetime
from dpy_jax import jax_functions_dpy as jf
from plotter import postplotter

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
dpy_dir = f"{scratch_dir}/dpy_jax"
qdpy_dir = f"{scratch_dir}/qdpy_jax"
outdir = f"{scratch_dir}/hybrid_jax"

def select_and_load():
    os.system(f"ls {outdir}/summary-* > {outdir}/fnames.txt")
    with open(f"{outdir}/fnames.txt", "r") as f:
        fnames = f.read().splitlines()

    for i in range(len(fnames)):
        print(f"{i:^5d} | {fnames[i]}")

    selector = int(input("Enter the index for filename: "))
    summary = jf.load_obj(f"{fnames[selector][:-4]}")
    return summary

summary = select_and_load()
GVARS = summary['params']['dpy']['GVARS']

c_arr_fit = summary['c_arr_fit']
true_params_flat = summary['true_params_flat']
cind_arr = summary['cind_arr']
sind_arr = summary['sind_arr']

suffix = f"{int(ARGS.knot_num)}s.{GVARS.eigtype}.{GVARS.tslen}d"
c_arr_fit_full = jf.c4fit_2_c4plot(GVARS, c_arr_fit*true_params_flat,
                                   sind_arr, cind_arr)
fit_plot = postplotter.postplotter(GVARS, c_arr_fit_full, f'summary-{suffix}')
