import argparse
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------------------------#
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
ARGS = PARSER.parse_args()
del PARSER
#-----------------------------------------------------------------------#
INSTR = ARGS.instrument


def reformat_splitdata():
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
            print(f"{n1:3d}, {ell1:4d},", idxs)
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


if __name__ == "__main__":
    a = np.loadtxt(f'splittings.out.{instrument}')
    ac_obs = a[:, 6]
    sorted_idx = np.argsort(ac_obs)

    ell = a[:, 0][sorted_idx].astype('int')
    n = a[:, 1][sorted_idx].astype('int')
    mu = a[:, 2][sorted_idx]
    sind = a[:, 3][sorted_idx].astype('int')

    ac_obs = a[:, 6][sorted_idx]
    ac_inv = a[:, 7][sorted_idx]
    asig = a[:, -1][sorted_idx]

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

    dsplits, dsplits_out = reformat_splitdata()
    np.savetxt(f'{instrument}.in.6335.36', dsplits)
    np.savetxt(f'{instrument}.out.6335.36', dsplits_out)

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