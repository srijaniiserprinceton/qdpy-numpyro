import numpy as np

NAX = np.newaxis

# loading the true parameters
true_params = np.load('true_params.npy')

# loading the exact supmat files
fixed_part = np.load('fixed_part.npy')
param_coeff = np.load('param_coeff.npy')
freq_diag = np.load('freq_diag.npy')

# loading the M files
fixed_part_M = np.load('fixed_part_M.npy')
param_coeff_M = np.load('param_coeff_M.npy')
p_dom_dell = np.load('p_dom_dell.npy')

# generating the exact supmat Z
Z = np.sum(param_coeff * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part + freq_diag

# generating the 0th order Z
Z0 = np.sum(param_coeff_M * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part_M + p_dom_dell
