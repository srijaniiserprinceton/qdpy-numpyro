import numpy as np
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

NAX = np.newaxis

#------------------------------------------------------------------------# 
true_params_flat = np.load('true_params_flat.npy')
model_params_sigma = np.abs(np.load('model_params_sigma.npy'))

#-----------------generating distribution of true params------------------------# 
num_params = len(true_params_flat)
num_samples = int(1e3)
true_params_samples = np.zeros((num_params, num_samples))

# looping over model params
for i in range(num_params):
    true_params_samples[i, :] = np.random.normal(loc=true_params_flat[i],
                                                 scale=model_params_sigma[i],
                                                 size=num_samples)

fig, ax = plt.subplots(2, 3, figsize=(10,8))
chosen_indices = np.array([0,num_params//2, -2])

for i in range(len(chosen_indices)):
    for j in range(2):
        param_ind = chosen_indices[i]
        data = true_params_samples[param_ind+j]
        # plotting the crude histograms
        (mu, sigma) = norm.fit(data)
        # the histogram of the data                                                         
        n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)
        
        # add a 'best fit' line                                                              
        y = norm.pdf(bins, mu, sigma)
        # ax[j,i].plot(bins, y, 'r--', linewidth=2)
        ax[j,i].grid(True)
        
plt.tight_layout()

fig.suptitle('Crude $m$', size=16)
fig.subplots_adjust(top=0.92)

plt.savefig('crude_params.png')

#-----------------finding the renormalized distributions------------------------# 
true_params_samples_renormed = true_params_samples/true_params_flat[:,NAX] - 1

fig, ax = plt.subplots(2, 3, figsize=(10,8))

for i in range(len(chosen_indices)):
    for j in range(2):
        param_ind = chosen_indices[i]
        data = true_params_samples_renormed[param_ind+j]
        # plotting the crude histograms                                                       
        (mu, sigma) = norm.fit(data)
        # the histogram of the data                                                           
        n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)

        # add a 'best fit' line                                                               
        y = norm.pdf(bins, mu, sigma)
        # ax[j,i].plot(bins, y, 'r--', linewidth=2)                                           
        ax[j,i].grid(True)

plt.tight_layout()

fig.suptitle('Step 1 normalization: $\\frac{m-m_0}{m_0}$', size=16)
fig.subplots_adjust(top=0.92)

plt.savefig('renormed_params.png')

#-------------sigma scaling the renormalized distributions------------------------# 
# array to store the sigma values to rescale renormed model params
sigma2scale = np.zeros(num_params)

for i in range(num_params):
    __, sigma2scale[i] = norm.fit(true_params_samples_renormed[i])

true_params_samples_final = true_params_samples_renormed/sigma2scale[:,NAX]

fig, ax = plt.subplots(2, 3, figsize=(10,8))

for i in range(len(chosen_indices)):
    for j in range(2):
        param_ind = chosen_indices[i]
        data = true_params_samples_final[param_ind+j]
        # plotting the crude histograms                                                       
        (mu, sigma) = norm.fit(data)
        # the histogram of the data                                                           
        n, bins, patches = ax[j,i].hist(data, 40, facecolor='green', alpha=0.75)

        # add a 'best fit' line                                                               
        y = norm.pdf(bins, mu, sigma)
        # ax[j,i].plot(bins, y, 'r--', linewidth=2)                                           
        ax[j,i].grid(True)

plt.tight_layout()

fig.suptitle('Step 2 normalization: $\\frac{m-m_0}{m_0}/\sigma$', size=16)
fig.subplots_adjust(top=0.92)

plt.savefig('final_params.png')
