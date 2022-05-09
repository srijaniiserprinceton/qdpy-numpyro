import numpy as np
import argparse
from preprocess import jsoc_params as jsp
import os
import sys
import time

#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
INSTR = dirnames[-1]
ipdir = f"{scratch_dir}/input_files"
instrdir = f"{ipdir}/{INSTR}"
dldir = f"{instrdir}/dlfiles"
splitdir = f"{dldir}/splitdir"
#-----------------------------------------------------------------------#
try:
    with open(f"{current_dir}/.jsoc_config", "r") as f:
        jsoc_config = f.read().splitlines()
    user_email = jsoc_config[0]
except FileNotFoundError:
    print(f"Please enter JSOC registered email in {current_dir}/.jsoc_config")
    sys.exit()
#-----------------------------------------------------------------------#
def retain_files(dirname, fname_filter):
    os.system(f"cd {dirname}; rm $(ls | egrep -v '{fname_filter}')")
    print(f"Retaining files: {dirname}/*{fname_filter}*")

params = jsp.jsocParams(instr=INSTR)
retain_files(dldir, f"{params.NDT}.36")
retain_files(splitdir, f"\\.{params.NDT}\\.")
retain_files(splitdir, "\\.36")
# os.system(f"cd {dldir}; rm $(ls | egrep -v '{NDT}.36')")
# os.system(f"cd {splitdir}; rm $(ls | egrep -v '.{NDT}.')")
# os.system(f"cd {splitdir}; rm $(ls | egrep -v '\\.36')")
