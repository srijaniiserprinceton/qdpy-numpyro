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

#-----------------------------------------------------------#
filenamepath = os.path.realpath(__file__)
configpath = '/'.join(filenamepath.split('/')[:-4])
with open(f"{configpath}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
#-----------------------------------------------------------#

# instrument directory in scratch storing processed files from JSOC
instr_ipfiles_dir = f'{scratch_dir}/input_files/{instr}'
datafnames = fnmatch.filter(os.listdir(instr_ipfiles_dir), f'{instr}.in.*')

# list to store the available daynum                                                        
data_daynum_list = []
for i in range(len(datafnames)):
    daynum_label = re.split('[.]+', datafnames[i], flags=re.IGNORECASE)[3]
    data_daynum_list.append(int(daynum_label))

data_daynum_arr = np.asarray(data_daynum_list)

smax = 9
# histogram params
bins = 20

# transparancy
alpha_arr = np.array([0.2, 0.3, 0.5])
color_arr = np.array(['k', 'b', 'r'])

# min ell array for filtering modes
min_ell_arr = np.array([0, 100, 200], dtype='int')
max_ell_arr = np.array([100, 200, 300], dtype='int')

# radial grid
r = np.loadtxt(f'{scratch_dir}/input_files/r.dat')

def_density = False

# for dimensionalization of wsr before plotting
M_sol = 1.989e33       # in grams                                                   
R_sol = 6.956e10       # in cm                                                               
B_0 = 10e5             # in Gauss (base of convection zone)                                  
OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)

def get_text_loc(ax):
    x_norm, y_norm = 0.05, 0.9
    if(ax == ax1 or ax == ax3):
        x_norm = 0.75
    
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    
    x_loc = x_norm * (x1 - x0) + x0
    y_loc = y_norm * (y1 - y0) + y0
    
    return x_loc, y_loc


for i in range(len(data_daynum_arr)): 
    daynum = data_daynum_arr[i]
    print(f'Processing {instr} day {daynum}.')
    
    modedata = np.loadtxt(f'{instr_ipfiles_dir}/{instr}.in.72d.{daynum}.36')
    
    # loading the multiplet array
    nl_arr = modedata[:, :2].astype('int')
    
    # masking 5 <= ell <= 300 and 0 <= n <= 30
    mask = (nl_arr[:,0] >= 5) * (nl_arr[:,0] <= 292) *\
           (nl_arr[:,1] >= 0) * (nl_arr[:,1] <= 30)
    
    modedata = modedata[mask]
    nl_arr = nl_arr[mask]
    
    # loading the acoeff_HMI array
    ac = modedata[:, 12:48][:,:smax+1:2].T
    
    # loading the acoeff_HMI sigma array
    ac_sigma = modedata[:,48:84][:,:smax+1:2].T
    
    # signal to noise ratio
    SNR = ac/ac_sigma
    
    # plot it
    fig = plt.figure(figsize=(15, 10))
    
    ax1= fig.add_subplot(2,3,1)
    ax3= fig.add_subplot(2,3,2)
    ax5= fig.add_subplot(2,3,3)
    
    ax7= fig.add_subplot(2,2,3)
    ax9= fig.add_subplot(2,2,4)
        
    for min_ell_ind, min_ell in enumerate(min_ell_arr):
        max_ell = max_ell_arr[min_ell_ind]
        mask_mult_1 = nl_arr[:,0] >= min_ell
        mask_mult_2 = nl_arr[:,0] < max_ell
        
        mask_mult = mask_mult_1 * mask_mult_2
        
        # creating the "new" histograms
        hist_1 = np.histogram(SNR[0, mask_mult], bins=bins, density=def_density)
        hist_3 = np.histogram(SNR[1, mask_mult], bins=bins, density=def_density)
        hist_5 = np.histogram(SNR[2, mask_mult], bins=bins, density=def_density)
        hist_7 = np.histogram(SNR[3, mask_mult], bins=bins, density=def_density)
        hist_9 = np.histogram(SNR[4, mask_mult], bins=bins, density=def_density)
        
        
        # plotting the histograms
        ax1.hist(hist_1[1][:-1], weights = hist_1[0], color=color_arr[min_ell_ind],
                 alpha=alpha_arr[min_ell_ind],
                 label='$%i \leq \ell < %i$'%(min_ell, max_ell))
        ax3.hist(hist_3[1][:-1], weights = hist_3[0], color=color_arr[min_ell_ind],
                 alpha=alpha_arr[min_ell_ind])
        ax5.hist(hist_5[1][:-1], weights = hist_5[0], color=color_arr[min_ell_ind],
                 alpha=alpha_arr[min_ell_ind])
        ax7.hist(hist_7[1][:-1], weights = hist_7[0], color=color_arr[min_ell_ind],
                 alpha=alpha_arr[min_ell_ind])
        ax9.hist(hist_9[1][:-1], weights = hist_9[0], color=color_arr[min_ell_ind],
                 alpha=alpha_arr[min_ell_ind])
    '''
    ax1.set_yticks([])
    ax3.set_yticks([])
    ax5.set_yticks([])
    ax7.set_yticks([])
    ax9.set_yticks([])
    '''
    # setting the texts in each subplot to indicate s
    x_text, y_text = get_text_loc(ax1)
    ax1.text(x_text, y_text, '$a_1/\sigma(a_1)$', fontsize=16)
    x_text, y_text = get_text_loc(ax3)
    ax3.text(x_text, y_text, '$a_3/\sigma(a_3)$', fontsize=16)
    x_text, y_text = get_text_loc(ax5)
    ax5.text(x_text, y_text, '$a_5/\sigma(a_5)$', fontsize=16)
    x_text, y_text = get_text_loc(ax7)
    ax7.text(x_text, y_text, '$a_7/\sigma(a_7)$', fontsize=16)
    x_text, y_text = get_text_loc(ax9)
    ax9.text(x_text, y_text, '$a_9/\sigma(a_9)$', fontsize=16)
    
    
    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower center', fontsize=18,
               ncol=4, borderaxespad=0.2)
    
    plt.suptitle("Normalized SNR histograms of odd $a$-coefficients", fontsize=18)
    
    plt.subplots_adjust(left=0.04,
                        bottom=0.1,
                        right=0.96,
                        top=0.94,
                        wspace=0.1,
                        hspace=0.1)
    
    plt.savefig(f'SNR_hist_{instr}_{daynum}.pdf')
    plt.close()
    
    # the wsr profiles plot
    fig = plt.figure(figsize=(17, 10))

    ax1= fig.add_subplot(2,3,1)
    ax3= fig.add_subplot(2,3,2)
    ax5= fig.add_subplot(2,3,3)

    ax7= fig.add_subplot(2,2,3)
    ax9= fig.add_subplot(2,2,4)
    
    wsr = np.loadtxt(f'{instr_ipfiles_dir}/wsr.{instr}.72d.{daynum}.{smax}') * -1.0
    # converting wsr to nhz
    wsr *= OM * 1e9

    ax1.plot(r, wsr[0], 'k', label='$w_1(r)$ in nHz')
    ax1.set_xlim([0, 1])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=16)
    
    ax3.plot(r, wsr[1], 'k', label='$w_3(r)$ in nHz')
    ax3.set_xlim([0, 1])
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=16)
    
    ax5.plot(r, wsr[2], 'k', label='$w_5(r)$ in nHz')
    ax5.set_xlim([0, 1])
    ax5.grid(alpha=0.3)
    ax5.legend(fontsize=16)
    
    ax7.plot(r, wsr[3], 'k', label='$w_7(r)$ in nHz')
    ax7.set_xlim([0, 1])
    ax7.grid(alpha=0.3)
    ax7.legend(fontsize=16)
    
    ax9.plot(r, wsr[4], 'k', label='$w_9(r)$ in nHz')
    ax9.set_xlim([0, 1])
    ax9.grid(alpha=0.3)
    ax9.legend(fontsize=16)
        
    plt.suptitle("1.5D $w_s(r)$ profiles from 2DRLS $\Omega(r,\\theta)$ JSOC profiles",
                 fontsize=18)

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
    
    plt.savefig(f'wsr_{instr}_{daynum}.pdf')
    plt.close()
    
    
