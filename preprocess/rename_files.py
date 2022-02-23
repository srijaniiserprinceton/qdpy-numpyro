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
#----------------------------------------------------------------------#
def rename_file(fname, suffix="split"):
    date = fname.split('.')[2].split('_')[0]
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    date_str = f"{year}-{month}-{day}"
    try:
        idx = np.where(date_str == mdi_daylist['DATE'].values)[0][0]
        print(f"{date_str} -- {mdi_daylist['DATE'][idx]} -- " +
              f"{mdi_daylist['MDI'][idx]}")
        mdi_day = mdi_daylist['MDI'][idx]
        found = 1
        os.system(f"cp {fname} {outdir}/hmi.{suffix}.{mdi_day}.18")
    except IndexError:
        print(f"{date_str} -- NOT FOUND")
        found = 0

    return found


def get_fnames(suffix="split"):
    os.system(f"ls {outdir}/hmi* | grep {suffix} > {outdir}/fnames_{suffix}.txt")
    with open(f"{outdir}/fnames_{suffix}.txt", "r") as f:
        fnames = f.read().splitlines()
    return fnames

if __name__ == "__main__":
    fnames_split = get_fnames("split")
    fnames_rot2d = get_fnames("rot")
    fnames_err2d = get_fnames("err")

    count = 0
    for fname in fnames_split:
        count += rename_file(fname, suffix="split")

    for fname in fnames_rot2d:
        count += rename_file(fname, suffix="rot2d")

    for fname in fnames_err2d:
        count += rename_file(fname, suffix="err2d")

    count = count // 3

    print(f"Total number of data chunks = {count}")
    print(f"Number of years = {count//5}")
