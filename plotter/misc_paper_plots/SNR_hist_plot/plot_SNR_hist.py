import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 

# loading the multiplet array
nl_arr = np.load('nl.0_30.5_292.6328.36.npy')

# loading the acoeff_HMI array
ac = np.load('acoeffs_HMI.hmi.72d.6328.36.npy')
ac = np.reshape(ac, (5, -1), 'F')

# loading the acoeff_HMI sigma array
ac_sigma = np.load('acoeffs_sigma_HMI.hmi.72d.6328.36.npy')
ac_sigma = np.reshape(ac_sigma, (5, -1), 'F')

# signal to noise ratio
SNR = ac/ac_sigma

# plot it
fig = plt.figure(figsize=(15, 10))

ax1= fig.add_subplot(2,3,1)
ax3= fig.add_subplot(2,3,2)
ax5= fig.add_subplot(2,3,3)

ax7= fig.add_subplot(2,2,3)
ax9= fig.add_subplot(2,2,4)

# min ell array for filtering modes
min_ell_arr = np.array([0, 100, 200], dtype='int')
max_ell_arr = np.array([100, 200, 300], dtype='int')

# histogram params
bins = 20

# transparancy
alpha_arr = np.array([0.2, 0.3, 0.5])
color_arr = np.array(['k', 'b', 'r'])

'''
mask_mult = nl_arr[:,1] >= min_ell_arr[0]
# creating the "old" histograms (from the previous ell_min)                                   
hist_1_old = np.histogram(SNR[0, mask_mult], bins=bins, density=True)
hist_3_old = np.histogram(SNR[1, mask_mult], bins=bins, density=True)
hist_5_old = np.histogram(SNR[2, mask_mult], bins=bins, density=True)
hist_7_old = np.histogram(SNR[3, mask_mult], bins=bins, density=True)
hist_9_old = np.histogram(SNR[4, mask_mult], bins=bins, density=True)
'''

for min_ell_ind, min_ell in enumerate(min_ell_arr):
    max_ell = max_ell_arr[min_ell_ind]
    mask_mult_1 = nl_arr[:,1] >= min_ell
    mask_mult_2 = nl_arr[:,1] < max_ell
    
    mask_mult = mask_mult_1 * mask_mult_2

    # creating the "new" histograms
    hist_1 = np.histogram(SNR[0, mask_mult], bins=bins, density=True)
    hist_3 = np.histogram(SNR[1, mask_mult], bins=bins, density=True)
    hist_5 = np.histogram(SNR[2, mask_mult], bins=bins, density=True)
    hist_7 = np.histogram(SNR[3, mask_mult], bins=bins, density=True)
    hist_9 = np.histogram(SNR[4, mask_mult], bins=bins, density=True)
    
    '''
    # calculating the weightfactor for each
    max_arg_h1 = np.argmax(hist_1[0])
    arg_h1_old = np.argmin(np.abs(hist_1_old[1] - hist_1[1][max_arg_h1]))
    w_h1 = hist_1_old[0][arg_h1_old] / hist_1[0][max_arg_h1]

    max_arg_h3 = np.argmax(hist_3[0])
    arg_h3_old = np.argmin(np.abs(hist_3_old[1] - hist_3[1][max_arg_h3]))
    w_h3 = hist_3_old[0][arg_h3_old] / hist_3[0][max_arg_h3]
    
    max_arg_h5 = np.argmax(hist_5[0])
    arg_h5_old = np.argmin(np.abs(hist_5_old[1] - hist_5[1][max_arg_h5]))
    w_h5 = hist_5_old[0][arg_h5_old] / hist_5[0][max_arg_h5]
    
    max_arg_h7 = np.argmax(hist_7[0])
    arg_h7_old = np.argmin(np.abs(hist_7_old[1] - hist_7[1][max_arg_h7]))
    w_h7 = hist_7_old[0][arg_h7_old] / hist_7[0][max_arg_h7]
    
    max_arg_h9 = np.argmax(hist_9[0])
    arg_h9_old = np.argmin(np.abs(hist_9_old[1] - hist_9[1][max_arg_h9]))
    w_h9 = hist_9_old[0][arg_h9_old] / hist_9[0][max_arg_h9]
    '''
    
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
    
    ax1.set_yticks([])
    ax3.set_yticks([])
    ax5.set_yticks([])
    ax7.set_yticks([])
    ax9.set_yticks([])

    '''
    # resetting the old histograms 
    hist_1_old = hist_1
    hist_3_old = hist_3
    hist_5_old = hist_5
    hist_7_old = hist_7
    hist_9_old = hist_9
    '''

def get_text_loc(ax):
    x_norm, y_norm = 0.05, 0.9
    if(ax == ax1 or ax == ax3):
        x_norm = 0.75
    
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    
    x_loc = x_norm * (x1 - x0) + x0
    y_loc = y_norm * (y1 - y0) + y0
    
    return x_loc, y_loc

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

plt.subplots_adjust(left=0.02,
                    bottom=0.1,
                    right=0.94,
                    top=0.94,
                    wspace=0.1,
                    hspace=0.1)

plt.savefig('SNR_hist.pdf')
plt.close()
