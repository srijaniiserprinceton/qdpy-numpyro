import os
import re
import sys
import argparse
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

parser = argparse.ArgumentParser()
parser.add_argument("--instr", help="hmi or mdi",
                    type=str, default="hmi")
PARGS = parser.parse_args()
instr = PARGS.instr

# run this code sitting in the directory of this code
dpy_dir = f'wsr-dpy'
hybrid_dir = f'wsr-hybrid'
jsoc_dir = f'wsr-jsoc'
wsr_sigma_dir = f'sigma-files'

plot_dir = 'wsr_plotfiles'

datafnames = fnmatch.filter(os.listdir(dpy_dir), f'wsr_dpy_fit_*.npy')

# list to store the available daynum                                                     
data_daynum_list = []
for i in range(len(datafnames)):
    daynum_label = re.split('[_,.]+', datafnames[i], flags=re.IGNORECASE)[3]
    data_daynum_list.append(int(daynum_label))

# extracted all the daynums
data_daynum_arr = np.asarray(data_daynum_list)

data_daynum_arr = np.array(['1288'])

# radial grid                                                                                 
r = np.load(f'r.npy')

# for dimensionalization of wsr before plotting                                              
OM = np.load('GVARS_OM.npy')

xlim_lo = 0.85
vline = 0.9

r_lo_ind = np.argmin(np.abs(r - xlim_lo))
r_up_ind = np.argmin(np.abs(r - 1))

def plot_two_wsr(wsr1, wsr2, wsr_sigma, type1, type2, day):
    # the wsr profiles plot                                                                
    fig = plt.figure(figsize=(17, 10))

    ax1= fig.add_subplot(2,3,1)
    ax3= fig.add_subplot(2,3,2)
    ax5= fig.add_subplot(2,3,3)

    ax7= fig.add_subplot(2,2,3)
    ax9= fig.add_subplot(2,2,4)

    axs = [ax1, ax3, ax5, ax7, ax9]

    def plot_s(ax, axnum):
        s = 2*axnum + 1
        #ax.plot(r[r_lo_ind:r_up_ind],
        #        wsr1[axnum,r_lo_ind:r_up_ind]-wsr2[axnum,r_lo_ind:r_up_ind],
        #        'k', label=f'{type1} $w_{s}(r)$ in nHz')
        ax.plot(r, wsr1[axnum], 'r', label=f'{type1} $w_{s}(r)$ in nHz')
        ax.plot(r, wsr2[axnum], 'r', label=f'{type2} $w_{s}(r)$ in nHz')
        ax.set_xlim([xlim_lo, 1])
        ax.axvline(vline)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=16)
        ax.fill_between(r[r_lo_ind:r_up_ind], -0.5*wsr_sigma[axnum, r_lo_ind:r_up_ind],
                        0.5*wsr_sigma[axnum, r_lo_ind:r_up_ind], color='b', alpha=0.4)

    for axnum in range(5):
        plot_s(axs[axnum], axnum)

    # plt.suptitle("1.5D $w_s(r)$ profiles from 2DRLS $\Omega(r,\\theta)$ JSOC profiles",
    #               fontsize=18)

    plt.subplots_adjust(left=0.06,
                        bottom=0.1,
                        right=0.98,
                        top=0.94,
                        wspace=0.12,
                        hspace=0.12)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes                                                 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('$r$ in $R_{\odot}$', fontsize=16)

    plt.savefig(f'{plot_dir}/wsr_{instr}_{type1}_{type2}_{day}.pdf')
    plt.close()


# next we plot the dpt vs jsoc and then dpt vs hybrid
for day in data_daynum_arr:
    wsr_jsoc = -1. * np.loadtxt(f'{jsoc_dir}/wsr.{instr}.72d.{day}.9')[:, 1:-1]
    wsr_dpy = np.load(f'{dpy_dir}/wsr_dpy_fit_{day}.npy')
    wsr_hybrid = np.load(f'{hybrid_dir}/wsr_hybrid_fit_{day}.npy')
    
    wsr_jsoc *= OM * 1e9
    wsr_dpy *= OM * 1e9
    wsr_hybrid *= OM * 1e9

    # loading the error from RLS
    wsr_sigma = np.loadtxt(f'{wsr_sigma_dir}/wsr_sigma_{day}.npy') * OM * 1e9
    
    plot_two_wsr(wsr_jsoc, wsr_dpy, wsr_sigma, 'JSOC', 'DPT', day)
    plot_two_wsr(wsr_hybrid, wsr_dpy, wsr_sigma, 'hybrid', 'DPT', day)
    
