import numpy as np
import subprocess
import os

#-----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
smax_global = int(dirnames[3])

dpy_batchdir = f"{scratch_dir}/batch_runs_dpy"
hybrid_batchdir = f"{scratch_dir}/batch_runs_hybrid"
muvalsmdi_dir = f"{scratch_dir}/muvals_mdi"
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
    knee_mu = []
    
    os.system(f'mkdir {muvalsmdi_dir}/{dpyname}')
    for s in range(1, smax_global+1, 2):
        try:
            knee_mu.append(np.load(f"{dpy_batchdir}/{dpyname}/muval.s{s}.npy"))
            os.system(f'cp {dpy_batchdir}/{dpyname}/muval.s{s}.npy' + 
                      f' {muvalsmdi_dir}/{dpyname}/.')
        except FileNotFoundError:
            knee_mu.append(1.0)
    knee_mu = np.asarray(knee_mu)
    print_str = f"{daynum:^6d} | "
    for i in range(len(knee_mu)):
        print_str = print_str + f" {knee_mu[i]:10.3e} |"
    print(print_str)
