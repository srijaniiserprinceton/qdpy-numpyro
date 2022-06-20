import numpy as np
import subprocess
import os

#-----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
write_dir = f"{scratch_dir}/organized-files"
hybrid_batchdir = f"{scratch_dir}/batch_runs_hybrid"
#-----------------------------------------------------------------------
def makedir(dirname):
    direxists = os.path.isdir(dirname)
    if not direxists: os.system(f"mkdir {dirname}")
    return None

def copy_files(srcname, destname):
    try:
        os.system(f"cp {srcname} {destname}")
    except FileNotFoundError:
        return None
    print(f"Writing {destname}")
    return None

makedir(write_dir)
makedir(f"{write_dir}/dpy-summary")
makedir(f"{write_dir}/hybrid-summary")
makedir(f"{write_dir}/sigma-files")
makedir(f"{write_dir}/acoeff-sigma-files")

print("====================================================================")
print(f"Collecting the summary file of dpy fit")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for hybridname in dirnames[:-1]:
    hybridsplit = hybridname.split('_')
    daynum = int(hybridsplit[2])
    os.system(f'cd {hybrid_batchdir}/{hybridname}/dpy_full_hess/summaryfiles; ls -arth summary* | ' +
              f'tail -1 > {hybrid_batchdir}/{hybridname}/dpy_full_hess/summaryfiles/latest_summary.txt')
    with open(f'{hybrid_batchdir}/{hybridname}/dpy_full_hess/summaryfiles/latest_summary.txt') as f:
        sumname = f.read().splitlines()
    print(sumname)
    copy_files(f"{hybrid_batchdir}/{hybridname}/dpy_full_hess/summaryfiles/{sumname[0]}",
               f"{write_dir}/dpy-summary/summary_{daynum}.pkl")

print("====================================================================")
print(f"Collecting the wsr sigma file")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for dpyname in dirnames[:-1]:
    dpysplit = dpyname.split('_')
    daynum = int(dpysplit[2])
    copy_files(f"{hybrid_batchdir}/{dpyname}/dpy_full_hess/wsr_sigma.npy",
               f"{write_dir}/sigma-files/wsr_sigma_{daynum}.npy")

print("====================================================================")
print(f"Collecting the acoeff sigma file of")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for hybridname in dirnames[:-1]:
    hybridsplit = hybridname.split('_')
    daynum = int(hybridsplit[2])
    copy_files(f"{hybrid_batchdir}/{hybridname}/dpy_full_hess/acoeffs_sigma_HMI*.npy",
               f"{write_dir}/acoeff-sigma-files/acoeff_sigma_{daynum}.npy")

print("====================================================================")
print(f"Collecting the summary file of hybrid fit")
dirnames = subprocess.check_output(["ls", f"{hybrid_batchdir}"])
dirnames = dirnames.decode().split('\n')
for hybridname in dirnames[:-1]:
    hybridsplit = hybridname.split('_')
    daynum = int(hybridsplit[2])
    copy_files(f"{hybrid_batchdir}/{hybridname}/summaryfiles/summary*.pkl",
               f"{write_dir}/hybrid-summary/summary_{daynum}.pkl")
