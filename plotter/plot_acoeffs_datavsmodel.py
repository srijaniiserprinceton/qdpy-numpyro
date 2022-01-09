import numpy as np
import matplotlib.pyplot as plt

def plot_acoeffs_datavsmodel(pred_acoeffs, data_acoeffs, data_acoeffs_sigma,
                             label):
    pred_acoeffs_plot = np.reshape(pred_acoeffs, (3,-1), 'F')
    data_acoeffs_plot = np.reshape(data_acoeffs, (3,-1), 'F')
    data_acoeffs_error = np.reshape(data_acoeffs_sigma, (3, -1), 'F')
    
    for i in range(3):
        plt.figure()
        plt.errorbar(np.arange(len(data_acoeffs_plot[i])),
                     data_acoeffs_plot[i], yerr=data_acoeffs_error[i], alpha=0.5,
                     fmt='o', ms=2, capsize=4, color='k')
        plt.plot(pred_acoeffs_plot[i], '.r', markersize=4)
        plt.savefig(f'a{2*i+1}_{label}.png')
        plt.close()
        
        plt.figure()
        plt.plot(pred_acoeffs_plot[i] - data_acoeffs_plot[i], '.k', markersize=4)
        plt.savefig(f'a{2*i+1}_res_{label}.png')
        plt.close()
