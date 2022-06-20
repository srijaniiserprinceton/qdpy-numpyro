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


def delete_files(fname):
    try:
        os.system(f"rm {fname}")
    except FileNotFoundError:
        print(f"{fname} NOT FOUND")
        return None
    print(f"Deleting {fname}")
    return None


print("====================================================================")
print(f"Collecting the carr_fit files from DPY fit")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for hybridname in dirnames[:-1]:
    hybridsplit = hybridname.split('_')
    daynum = int(hybridsplit[2])
    print(f"====================== {daynum} ==============================================")
    delete_files(f"{hybrid_batchdir}/{hybridname}/dpy_files/param_coeff*")
    delete_files(f"{hybrid_batchdir}/{hybridname}/qdpy_files/param_coeff*")
    delete_files(f"{hybrid_batchdir}/{hybridname}/qdpy_full_hess/param_coeff*")
    
    delete_files(f"{hybrid_batchdir}/{hybridname}/dpy_files/RL_poly*")
    delete_files(f"{hybrid_batchdir}/{hybridname}/qdpy_files/RL_poly*")
    delete_files(f"{hybrid_batchdir}/{hybridname}/qdpy_full_hess/RL_poly*")
