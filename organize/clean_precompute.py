import numpy as np
import subprocess
import os

#-----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
dpy_batchdir = f"{scratch_dir}/batch_runs_dpy"
hybrid_batchdir = f"{scratch_dir}/batch_runs_hybrid"
#-----------------------------------------------------------------------


def copy_files(srcname, destname):
    try:
        os.system(f"cp {srcname} {destname}")
    except FileNotFoundError:
        return None
    print(f"Writing {destname}")
    return None


print("====================================================================")
print(f"Collecting the carr_fit files from DPY fit")
dirnames = subprocess.check_output(["ls", f"{dpy_batchdir}"])
dirnames = dirnames.decode().split('\n')
for dpyname in dirnames[:-1]:
    dpysplit = dpyname.split('_')
    daynum = int(dpysplit[2])
    copy_files(f"{dpy_batchdir}/{dpyname}/carr_fit_1.00000e+00.npy",
               f"{scratch_dir}/carr_dpy/carr_dpy_{daynum}.npy")

print("====================================================================")
print(f"Collecting the summary file of hybrid fit")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for hybridname in dirnames[:-1]:
    hybridsplit = hybridname.split('_')
    daynum = int(hybridsplit[2])
    copy_files(f"{hybrid_batchdir}/{hybridname}/summaryfiles/summary*.pkl",
               f"{scratch_dir}/hybrid-summary/summary_{daynum}.pkl")

print("====================================================================")
print(f"Collecting the wsr sigma file")
dirnames = subprocess.check_output(["ls", f"{dpy_batchdir}"])
dirnames = dirnames.decode().split('\n')
for dpyname in dirnames[:-1]:
    dpysplit = dpyname.split('_')
    daynum = int(dpysplit[2])
    copy_files(f"cp {dpy_batchdir}/{dpyname}/wsr_sigma.npy",
               f"{scratch_dir}/sigma-files/wsr_sigma_{daynum}.npy")
