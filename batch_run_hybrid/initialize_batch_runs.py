import os
import shutil
import numpy as np
import fnmatch
import re
import sys
import argparse

import make_run_params

#-------------------------------------------------------------------#
parser = argparse.ArgumentParser()
ARGS = parser.parse_args()

#--------------------------------------------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
datadir = f"{scratch_dir}/input_files/hmi"

#------------------READING THE DATA FILE NAMES-----------------------#
datafnames = fnmatch.filter(os.listdir(datadir), 'hmi.in.*')

# list to store the available daynum
data_daynum_list = []

for i in range(len(datafnames)):
    daynum_label = re.split('[.]+', datafnames[i], flags=re.IGNORECASE)[3]
    data_daynum_list.append(int(daynum_label))

#---------------CREATING THE BATCHRUN STRUCTURE-----------------------#
# the directory which contains subdirectories of the batch run
batch_run_dir = f"{scratch_dir}/batch_runs_hybrid"

for i in range(len(data_daynum_list)):
    daynum = data_daynum_list[i]
    batch_subdirpath = f"{batch_run_dir}/hmi_72d_{daynum}_18" 

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
    os.mkdir(f"{batch_subdirpath}/qdpy_files")
    os.mkdir(f"{batch_subdirpath}/dpy_full_hess")


nmin, nmax, lmin, lmax = 0, 0, 200, 210
nmin_q, nmax_q, lmin_q, lmax_q = 0, 0, 200, 201

# writing the parameters in each run directory for bookkeeping
for i in range(len(data_daynum_list)):
    daynum = data_daynum_list[i]
    run_dir_dpy = f"{batch_run_dir}/hmi_72d_{daynum}_18/dpy_files"
    run_dir_qdpy = f"{batch_run_dir}/hmi_72d_{daynum}_18/qdpy_files"
    run_dir_dpy_full_hess = f"{batch_run_dir}/hmi_72d_{daynum}_18/dpy_full_hess"

    #--------------------------------------------------------------------#
    # making dpy params file for full modeset for hessian                           
    run_params = make_run_params.make_run_params(nmin=nmin,nmax=nmax,
                                                 lmin=lmin,lmax=lmax,
                                                 daynum=daynum)
    smin, smax = run_params.smin, run_params.smax
    with open(f"{run_dir_dpy_full_hess}/.params_smin_{smin}_smax_{smax}.dat", "w") as f:
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
                f"{run_params.numsplit}" + "\n"+
                f"{run_params.exclude_qdpy}")

    #--------------------------------------------------------------------#
    # making qdpy params file
    run_params = make_run_params.make_run_params(nmin=nmin_q,nmax=nmax_q,
                                                 lmin=lmin_q,lmax=lmax_q,
                                                 daynum=daynum)
    smin, smax = run_params.smin, run_params.smax
    with open(f"{run_dir_qdpy}/.params_smin_{smin}_smax_{smax}.dat", "w") as f:
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
                f"{run_params.numsplit}" + "\n"+
                f"{run_params.exclude_qdpy}")

    #--------------------------------------------------------------------#
    # making dpy params file
    run_params = make_run_params.make_run_params(nmin=nmin,nmax=nmax,
                                                 lmin=lmin,lmax=lmax,
                                                 daynum=daynum,
                                                 exclude_qdpy=1)
    smin, smax = run_params.smin, run_params.smax
    with open(f"{run_dir_dpy}/.params_smin_{smin}_smax_{smax}.dat", "w") as f:
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
                f"{run_params.numsplit}" + "\n"+
                f"{run_params.exclude_qdpy}")
        
    #--------------------------------------------------------------------#
