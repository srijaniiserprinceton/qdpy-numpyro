import subprocess
import numpy as np
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

_pythonpath = subprocess.check_output("which python",
                                        shell=True)
pythonpath = _pythonpath.decode("utf-8").split("\n")[0]
batch_run_dir = f"{scratch_dir}/batch_runs_hybrid"

batchnames = [filename for
              filename in os.listdir(f"{batch_run_dir}")
              if (os.path.isdir(f"{scratch_dir}/batch_runs_hybrid/{filename}")
                  and filename[0]!='.')]

for bname in batchnames:
    os.system(f"cp {batch_run_dir}/{bname}/dpy_files/summaryfiles/*.pkl " +
              f"{batch_run_dir}/{bname}/summaryfiles/")
    os.system(f"cp {batch_run_dir}/{bname}/qdpy_files/summaryfiles/*.pkl " +
              f"{batch_run_dir}/{bname}/summaryfiles/")
    os.system(f"cp {batch_run_dir}/{bname}/dpy_full_hess/summaryfiles/*.pkl " +
              f"{batch_run_dir}/{bname}/summaryfiles/")
