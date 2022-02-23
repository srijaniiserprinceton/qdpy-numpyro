import numpy as np
import pandas as pd
import os

#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
outdir = f"{ipdir}/hmi"
#----------------------------------------------------------------------#

mdi_daylist = pd.read_table(f'{ipdir}/daylist.txt', delim_whitespace=True,
                            names=('SN', 'MDI', 'DATE'),
                            dtype={'SN': np.int64,
                                   'MDI': np.int64,
                                   'DATE': str})

def rename_file(fname):
    date = fname.split('.')[2].split('_')[0]
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    date_str = f"{year}-{month}-{day}"
    try:
        idx = np.where(date_str == mdi_daylist['DATE'].values)[0][0]
        print(f"{date_str} -- {mdi_daylist['DATE'][idx]} -- " +
              f"{mdi_daylist['MDI'][idx]}")
        found = 1
    except IndexError:
        print(f"{date_str} -- NOT FOUND")
        found = 0

    return found


if __name__ == "__main__":
    os.system(f"ls {outdir}/hmi* | grep splittings > {outdir}/fnames.txt")
    with open(f"{outdir}/fnames.txt", "r") as f:
        fnames = f.read().splitlines()

    count = 0
    for fname in fnames:
        count += rename_file(fname)

    print(f"Total number of data chunks = {count}")
    print(f"Number of years = {count//5}")
