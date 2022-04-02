import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from preprocess import rename_files as RN
#-----------------------------------------------------------------------#
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
PARSER.add_argument("--tslen", help="72d or 360d",
                    type=str, default="72d")
ARGS = PARSER.parse_args()
del PARSER
#-----------------------------------------------------------------------#
INSTR = ARGS.instrument
#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
dldir = f"{ipdir}/{INSTR}"
#----------------------------------------------------------------------#


def reformat_splitdata(ell, n, mu, sind, ac_ois):
    ac_obs, ac_inv, asig = ac_ois
    """Creates files in the format of hmi.6328.36"""
    num_modes = len(ell)
    data_splits = np.zeros((num_modes, 84), dtype=float)
    data_splits_out = np.zeros((num_modes, 84), dtype=float)
    count = 0
    ell_list = np.unique(ell)
    for ell1 in ell_list:
        ellidx = np.where(ell == ell1)[0]
        nlist = np.unique(n[ellidx])
        for n1 in nlist:
            idxs = np.where((ell1 == ell)*
                            (n1 == n))[0]
            # print(f"{n1:3d}, {ell1:4d},", idxs)
            data_splits[count, 49:49+36] = 1.0
            data_splits_out[count, 49:49+36] = 1.0
            for idx in idxs:
                data_splits[count, 0] = ell[idx]
                data_splits[count, 1] = n[idx]
                data_splits[count, 2] = mu[idx]
                data_splits[count, 11+2*sind[idx]-1] = ac_obs[idx]
                data_splits[count, 48+2*sind[idx]-1] = asig[idx]
                data_splits_out[count, 11+2*sind[idx]-1] = ac_inv[idx]
                data_splits_out[count, 48+2*sind[idx]-1] = asig[idx]
            count += 1

    data_splits_out[:, :11] = data_splits[:, :11]
    return data_splits, data_splits_out


def get_fnames(suffix="split"):
    os.system(f"ls {dldir}/dlfiles/hmi* | grep {suffix} > " +
              f"{dldir}/dlfiles/fnames_{suffix}.txt")
    with open(f"{dldir}/dlfiles/fnames_{suffix}.txt", "r") as f:
        fnames = f.read().splitlines()
    return fnames


def setup_reformatting(fname):
    a = np.loadtxt(fname)
    ac_obs = a[:, 6]
    sorted_idx = np.argsort(ac_obs)

    ell = a[:, 0][sorted_idx].astype('int')
    n = a[:, 1][sorted_idx].astype('int')
    mu = a[:, 2][sorted_idx]
    sind = a[:, 3][sorted_idx].astype('int')

    ac_inv = a[:, 6][sorted_idx]
    ac_obs = a[:, 7][sorted_idx]
    asig = a[:, -1][sorted_idx]
    ac_ois = (ac_obs, ac_inv, asig)

    dsplits, dsplits_out = reformat_splitdata(ell, n, mu, sind, ac_ois)
    return dsplits, dsplits_out


def plot_data():
    a1obs = ac_obs[sind==1]
    a1inv = ac_inv[sind==1]
    a1sig = asig[sind==1]
    ell1 = ell[sind==1]
    a1idx = np.arange(len(a1obs))

    a3obs = ac_obs[sind==2]
    a3inv = ac_inv[sind==2]
    a3sig = asig[sind==2]
    ell3 = ell[sind==2]
    a3idx = np.arange(len(a3obs))

    a5obs = ac_obs[sind==3]
    a5inv = ac_inv[sind==3]
    a5sig = asig[sind==3]
    ell5 = ell[sind==3]
    a5idx = np.arange(len(a5obs))

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
    #axs = axs.flatten()
    plot_xaxis = [a1idx, a3idx, a5idx]
    #plot_xaxis = [ell1, ell3, ell5]

    axs[0, 0].errorbar(plot_xaxis[0], a1obs, yerr=a1sig,
                       alpha=0.5, fmt='o', ms=2, capsize=4, color='k')
    axs[0, 0].plot(plot_xaxis[0], a1inv, '.r', markersize=4)
    axs[0, 0].set_xlabel('ell')
    axs[0, 0].set_ylabel('a1')

    axs[0, 1].plot(ell1, (a1inv-a1obs)/a1sig, '.k', markersize=4)
    axs[0, 1].set_xlabel('ell')
    axs[0, 1].set_ylabel('$\\delta a_1/\\sigma_1$')
    #--------------
    axs[1, 0].errorbar(plot_xaxis[1], a3obs, yerr=a3sig,
                       alpha=0.5, fmt='o', ms=2, capsize=4, color='k')
    axs[1, 0].plot(plot_xaxis[1], a3inv, '.r', markersize=4)
    axs[1, 0].set_xlabel('ell')
    axs[1, 0].set_ylabel('a3')

    axs[1, 1].plot(ell3, (a3inv-a3obs)/a3sig, '.k', markersize=4)
    # axs[1, 1].set_yscale('symlog')
    axs[1, 1].set_xlabel('ell')
    axs[1, 1].set_ylabel('$\\delta a_3/\\sigma_3$')
    #--------------
    axs[2, 0].errorbar(plot_xaxis[2], a5obs, yerr=a5sig,
                       alpha=0.5, fmt='o', ms=2, capsize=4, color='k')
    axs[2, 0].plot(plot_xaxis[2], a5inv, '.r', markersize=4)
    axs[2, 0].set_xlabel('ell')
    axs[2, 0].set_ylabel('a5')

    axs[2, 1].plot(ell5, (a5inv-a5obs)/a5sig, '.k', markersize=4)
    # axs[2, 1].set_yscale('symlog')
    axs[2, 1].set_xlabel('ell')
    axs[2, 1].set_ylabel('$\\delta a_5/\\sigma_5$')
    fig.tight_layout()
    plt.show()
    return None


def store_output(fname, splitdata):
    fmt_list = []
    for i in range(splitdata.shape[1]):
        if i<2:
            fmt_list.append("^5d")
        else:
            fmt_list.append(".4f")

    with open(fname, "w") as f:
        for i in range(splitdata.shape[0]):
            for j in range(splitdata.shape[1]):
                if j<2:
                    f.write(f"{int(splitdata[i, j]):{fmt_list[j]}} ")
                else:
                    f.write(f"{splitdata[i, j]:{fmt_list[j]}} ")
            f.write("\n")
    return None



if __name__ == "__main__":
    fnames_split = get_fnames()
    for fname in fnames_split:
        newname = RN.get_newname(fname)
        print(newname)
        dsplits, dsplits_out = setup_reformatting(fname)
        fname_splits = newname.split('.')
        mdi_day = fname_splits[2]
        numsplits = fname_splits[3]

        store_output(f'{dldir}/{INSTR}.in.{ARGS.tslen}.{mdi_day}.{numsplits}', dsplits)
        store_output(f'{dldir}/{INSTR}.out.{ARGS.tslen}.{mdi_day}.{numsplits}',
                     dsplits_out)
