import os
import shutil
import fnmatch
import numpy as np
import re
import sys

import make_run_params

#--------------------------------------------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
INSTR = dirnames[4]
smax_global = int(dirnames[3])

datadir = f"{scratch_dir}/input_files/{INSTR}"

#------------------READING THE DATA FILE NAMES-----------------------#
datafnames = fnmatch.filter(os.listdir(datadir), f'{INSTR}.in.*')

# list to store the available daynum
data_daynum_list = []
data_numsplit_list = []

for i in range(len(datafnames)):
    daynum_label = re.split('[.]+', datafnames[i], flags=re.IGNORECASE)[3]
    num_splits = re.split('[.]+', datafnames[i], flags=re.IGNORECASE)[-1]
    data_daynum_list.append(int(daynum_label))
    data_numsplit_list.append(int(num_splits))

#---------------CREATING THE BATCHRUN STRUCTURE-----------------------#
# the directory which contains subdirectories of the batch run
batch_run_dir = f"{scratch_dir}/batch_runs_hybrid"

for i in range(len(data_daynum_list)):
    daynum = data_daynum_list[i]
    numsplits = data_numsplit_list[i]
    batch_subdirpath = f"{batch_run_dir}/{INSTR}_72d_{daynum}_{numsplits}" 

    # creating the directory for one of the batch runs
    if(os.path.isdir(batch_subdirpath)):
        # deleting the directory if it already exists
        try:
            shutil.rmtree(batch_subdirpath)
        except OSError as e:
            print("Error: %s : %s" % (batch_subdirpath, e.strerror))
            
    # making new directory
    os.mkdir(batch_subdirpath)
    
    # making sub directories needed
    os.mkdir(f"{batch_subdirpath}/plots")
    os.mkdir(f"{batch_subdirpath}/summaryfiles")

    os.mkdir(f"{batch_subdirpath}/dpy_files")
    os.mkdir(f"{batch_subdirpath}/dpy_files/plots")
    os.mkdir(f"{batch_subdirpath}/dpy_files/summaryfiles")

    os.mkdir(f"{batch_subdirpath}/qdpy_files")
    os.mkdir(f"{batch_subdirpath}/qdpy_files/plots")
    os.mkdir(f"{batch_subdirpath}/qdpy_files/summaryfiles")

    os.mkdir(f"{batch_subdirpath}/dpy_full_hess")
    os.mkdir(f"{batch_subdirpath}/dpy_full_hess/plots")
    os.mkdir(f"{batch_subdirpath}/dpy_full_hess/summaryfiles")


def write_paramsfile(dirname, fname, run_params):
    with open(f"{dirname}/{fname}", "w") as f:
        f.write(f"{run_params.nmin}" + "\n" +
                f"{run_params.nmax}" + "\n" +
                f"{run_params.lmin}" + "\n" +
                f"{run_params.lmax}" + "\n" +
                f"{smin}" + "\n" +
                f"{smax}" + "\n" +
                f"{run_params.knotnum}" + "\n" +
                f"{run_params.rth}" + "\n" +
                f"{run_params.tslen}" + "\n" +
                f"{run_params.daynum}" + "\n" +
                f"{run_params.numsplit}" + "\n" +
                f"{run_params.exclude_qdpy}" + "\n" +
                f"{run_params.smax_global}")
    return None


# nmin, nmax, lmin, lmax = 0, 0, 200, 210
# nmin_q, nmax_q, lmin_q, lmax_q = 0, 0, 200, 202
nmin, nmax, lmin, lmax = 0, 30, 5, 292
nmin_q, nmax_q, lmin_q, lmax_q = 0, 30, 161, 292

smin, smax = 1, smax_global

# writing the parameters in each run directory for bookkeeping
for i in range(len(data_daynum_list)):
    daynum = data_daynum_list[i]
    numsplits = data_numsplit_list[i]
    rundir_dpy = f"{batch_run_dir}/hmi_72d_{daynum}_{numsplits}/dpy_files"
    rundir_qdpy = f"{batch_run_dir}/hmi_72d_{daynum}_{numsplits}/qdpy_files"
    rundir_dpy_full_hess = f"{batch_run_dir}/hmi_72d_{daynum}_{numsplits}/dpy_full_hess"

    #--------------------------------------------------------------------#
    # making dpy params file for full modeset for hessian                           
    fname = f".params_smin_{smin}_smax_{smax}.dat"
    
    # making dpy-full params file
    run_params = make_run_params.make_run_params(nmin=nmin, nmax=nmax,
                                                 lmin=lmin, lmax=lmax,
                                                 smin=smin, smax=smax,
                                                 smax_global=smax_global,
                                                 daynum=daynum)
    write_paramsfile(rundir_dpy_full_hess, fname, run_params)
    
    # making qdpy params file
    run_params = make_run_params.make_run_params(nmin=nmin_q, nmax=nmax_q,
                                                 lmin=lmin_q, lmax=lmax_q,
                                                 smin=smin, smax=smax,
                                                 smax_global=smax_global,
                                                 daynum=daynum)
    write_paramsfile(rundir_qdpy, fname, run_params)
    
    # making dpy params file
    run_params = make_run_params.make_run_params(nmin=nmin, nmax=nmax,
                                                 lmin=lmin, lmax=lmax,
                                                 smin=smin, smax=smax,
                                                 smax_global=smax_global,
                                                 daynum=daynum,
                                                 exclude_qdpy=1)
    write_paramsfile(rundir_dpy, fname, run_params)
