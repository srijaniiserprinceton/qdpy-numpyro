import os
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
batch_dir = f"{scratch_dir}/batch_runs_dpy"
bashscr_dir = f"{package_dir}/jobscripts/bashbatch"

#----------------- getting full pythonpath -----------------------
_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
#-----------------------------------------------------------------
batchnames = [filename for filename in os.listdir(batch_dir) if 
              (os.path.isdir(f"{batch_dir}/{filename}") and filename[0]!='.')]

for bname in batchnames:
    with open(f"{bashscr_dir}/{bname}.sh", "w") as f:
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py --rundir {batch_dir}/{bname} --s 1\n")
        f.write(f"{pythonpath} {current_dir}/batch_iterative_inversion.py --rundir {batch_dir}/{bname} --s 1\n")
        f.write(f"{pythonpath} {current_dir}/mu_bisection_batch.py --rundir {batch_dir}/{bname} --s 1\n")
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py --rundir {batch_dir}/{bname} --s 3\n")
        f.write(f"{pythonpath} {current_dir}/batch_iterative_inversion.py --rundir {batch_dir}/{bname} --s 3\n")
        f.write(f"{pythonpath} {current_dir}/mu_bisection_batch.py --rundir {batch_dir}/{bname} --s 3\n")
        f.write(f"{pythonpath} {current_dir}/batch_precompute.py --rundir {batch_dir}/{bname} --s 5\n")
        f.write(f"{pythonpath} {current_dir}/batch_iterative_inversion.py --rundir {batch_dir}/{bname} --s 5\n")
        f.write(f"{pythonpath} {current_dir}/mu_bisection_batch.py --rundir {batch_dir}/{bname} --s 5")
