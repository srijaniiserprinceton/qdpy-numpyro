import os
import shutil
import numpy as np
import fnmatch
import re
import sys

import make_run_params

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
batch_run_dir = f"{scratch_dir}/batch_runs_dpy"

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


# adding optional parameters to not use default

nmin, nmax, lmin, lmax = 0, 30, 5, 292
# nmin, nmax, lmin, lmax = 0, 0, 200, 210

smax_global = int(dirnames[3])

# at this point smin_arr = smax_arr = [1, 3, 5, ....]
smin_arr = np.arange(1, smax_global+1, 2)
smax_arr = np.arange(1, smax_global+1, 2)

# at this point smin_arr = [1, 3, 5, ..., 1] and smax_arr = [1, 3, 5, ..., smax_global]
smax = 5
smin_arr = np.append(smin_arr, 1)
smax_arr = np.append(smax_arr, smax)
# smax_arr = np.append(smax_arr, smax_global)

# writing the parameters in each run directory for bookkeeping
for i in range(len(data_daynum_list)):
    daynum = data_daynum_list[i]
    run_dir = f"{batch_run_dir}/hmi_72d_{daynum}_18"

    for j in range(len(smin_arr)):
        smin, smax = smin_arr[j], smax_arr[j]
        
        # getting dictionary of run params
        run_params = make_run_params.make_run_params(smin=smin,smax=smax,
                                                     nmin=nmin,nmax=nmax,
                                                     lmin=lmin,lmax=lmax,
                                                     smax_global=smax_global,
                                                     daynum=daynum)

        with open(f"{run_dir}/.params_smin_{smin}_smax_{smax}.dat", "w") as f:
            f.write(f"{run_params.nmin}" + "\n" +
                    f"{run_params.nmax}" + "\n" +
                    f"{run_params.lmin}" + "\n" +
                    f"{run_params.lmax}" + "\n" +
                    f"{run_params.smin}" + "\n" +
                    f"{run_params.smax}" + "\n" +
                    f"{run_params.knotnum}" + "\n" +
                    f"{run_params.rth}" + "\n" +
                    f"{run_params.tslen}" + "\n" +
                    f"{run_params.daynum}" + "\n" +
                    f"{run_params.numsplit}" + "\n" +
                    f"{run_params.exclude_qdpy}" + "\n" +
                    f"{run_params.smax_global}")
