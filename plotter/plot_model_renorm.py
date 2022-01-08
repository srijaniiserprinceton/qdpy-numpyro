import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def visualize_model_renorm(true_params_flat, true_params_samples, sigma,
                           renorm_fn, len_s):
    num_params = len(true_params_flat)
    
    # to facilitate fivision without using NAX
    true_params_flat_shaped = np.reshape(true_params_flat, (num_params, 1))
    sigma = np.reshape(sigma, (num_params, 1))
    
    #-----------------plotting the crude distributions------------------------# 
    fig, ax = plt.subplots(len_s, 3, figsize=(10,8))
    chosen_indices = np.array([0,num_params//2, -2])
    
    for i in range(len(chosen_indices)):
        for j in range(len_s):
            param_ind = chosen_indices[i]
            data = true_params_samples[param_ind+j]
                        
            # the histogram of the data                                                       
            n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)
            
            ax[j,i].grid(True)
            
    plt.tight_layout()
    
    fig.suptitle('Crude $m$', size=16)
    fig.subplots_adjust(top=0.92)
    
    plt.savefig('crude_params.png')
    
    #-----------------finding the renormalized distributions------------------------# 
    true_params_samples_renormed = renorm_fn(true_params_samples,
                                             true_params_flat_shaped,
                                             1.0)
    
    fig, ax = plt.subplots(2, 3, figsize=(10,8))
            
    for i in range(len(chosen_indices)):
        for j in range(len_s):
            param_ind = chosen_indices[i]
            data = true_params_samples_renormed[param_ind+j]
            
            # the histogram of the data                                                       
            n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)
            
            ax[j,i].grid(True)
            
    plt.tight_layout()
    
    fig.suptitle('Step 1 normalization: $\\frac{m-m_0}{m_0}$', size=16)
    # fig.suptitle('Step 1 normalization: $log(\\frac{m}{m_0})$', size=16)
    fig.subplots_adjust(top=0.92)
    
    plt.savefig('renormed_params.png')
    
    #-------------sigma scaling the renormalized distributions------------------------# 
    true_params_samples_final = true_params_samples_renormed/sigma
    
    fig, ax = plt.subplots(2, 3, figsize=(10,8))

    for i in range(len(chosen_indices)):
        for j in range(len_s):
            param_ind = chosen_indices[i]
            data = true_params_samples_final[param_ind+j]
                        
            # the histogram of the data                                                   
            n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)
            
            ax[j,i].grid(True)
            
    plt.tight_layout()
    
    fig.suptitle('Step 2 normalization: $\\frac{m-m_0}{m_0}/\sigma$', size=16)
    # fig.suptitle('Step 2 normalization: $log(\\frac{m}{m_0})/\sigma$', size=16)
    fig.subplots_adjust(top=0.92)
    
    plt.savefig('final_params.png')
