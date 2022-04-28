import os
import numpy as np
import matplotlib.pyplot as plt
from jax.ops import index as jidx
from jax.ops import index_update as jidx_update

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
plotdir = f"{scratch_dir}/plots"


def plot_acoeffs_datavsmodel(pred_acoeffs, data_acoeffs, data_acoeffs_out_HMI,
                             data_acoeffs_sigma, label, plotdir=plotdir, len_s=10):
    if pred_acoeffs.shape[0]%len_s != 0:
        len_s += 1
    
    pred_acoeffs_plot = np.reshape(pred_acoeffs, (len_s, -1), 'F')
    data_acoeffs_plot = np.reshape(data_acoeffs, (len_s, -1), 'F')
    data_acoeffs_out_HMI_plot = np.reshape(data_acoeffs_out_HMI,
                                           (len_s, -1), 'F')
    data_acoeffs_error = np.reshape(data_acoeffs_sigma, (len_s, -1), 'F')
    
    for i in range(len_s):
        plt.figure()
        plt.errorbar(np.arange(len(data_acoeffs_plot[i])),
                     data_acoeffs_plot[i], yerr=data_acoeffs_error[i], alpha=0.5,
                     fmt='o', ms=2, capsize=4, color='k')
        plt.plot(pred_acoeffs_plot[i], '.r', markersize=4)
        plt.plot(data_acoeffs_out_HMI_plot[i], 'xb', markersize=4)
        plt.savefig(f'{plotdir}/a{2*i+1}_{label}.png')
        plt.close()
        
        plt.figure()
        plt.plot(pred_acoeffs_plot[i] - data_acoeffs_plot[i], '.k', markersize=4)
        plt.savefig(f'{plotdir}/a{2*i+1}_res_{label}.png')
        plt.close()

def plot_acoeffs_dm_scaled(pred_acoeffs, data_acoeffs, data_acoeffs_out_HMI,
                           data_acoeffs_sigma, label, plotdir=plotdir, len_s=10):

    if pred_acoeffs.shape[0]%len_s != 0:
        len_s += 1
    pred_acoeffs_plot = np.reshape(pred_acoeffs, (len_s, -1), 'F')
    data_acoeffs_plot = np.reshape(data_acoeffs, (len_s, -1), 'F')
    data_acout_HMI_plot = np.reshape(data_acoeffs_out_HMI, (len_s, -1), 'F')
    data_acoeffs_error = np.reshape(data_acoeffs_sigma, (len_s, -1), 'F')

    pred_acoeffs_plot = (pred_acoeffs_plot - data_acoeffs_plot)/data_acoeffs_error
    data_acout_HMI_plot = (data_acout_HMI_plot - data_acoeffs_plot)/data_acoeffs_error
    data_acoeffs_plot = (data_acoeffs_plot - data_acoeffs_plot)/data_acoeffs_error
    
    for i in range(len_s):
        plt.figure()
        plt.errorbar(np.arange(len(data_acoeffs_plot[i])),
                     data_acoeffs_plot[i],
                     yerr=np.ones_like(data_acoeffs_plot[i]),
                     alpha=0.1, fmt='o', ms=2, capsize=4, color='k')
        plt.plot(data_acout_HMI_plot[i], 'xb', markersize=4)
        plt.plot(pred_acoeffs_plot[i], '.r', markersize=4)
        plt.savefig(f'{plotdir}/a{2*i+1}_scaled_{label}.png')
        plt.close()
        
        plt.figure()
        plt.plot(pred_acoeffs_plot[i] - data_acoeffs_plot[i], '.k', markersize=4)
        plt.savefig(f'{plotdir}/a{2*i+1}_scaledres_{label}.png')
        plt.close()

    for i in range(len_s):
        plt.figure()
        plt.plot(data_acout_HMI_plot[i],
                 pred_acoeffs_plot[i], '.k', markersize=2, alpha=0.5)
        plt.savefig(f'{plotdir}/a{2*i+1}_correlation.png')
        plt.close()
