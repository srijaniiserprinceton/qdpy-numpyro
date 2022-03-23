import numpy as np
import matplotlib.pyplot as plt

# loading the basis functions 
bsp_basis_full = np.load('bsp_basis_full.npy')
# loading the radial mesh
r = np.load('r.npy')

# plotting
fig, ax = plt.subplots(1, 1, figsize=(12,5))

r_th_ind = np.argmin(np.abs(r - 0.88))

for i in range(0, len(bsp_basis_full) - 60):
    ax.plot(r[r_th_ind:], bsp_basis_full[i, r_th_ind:], 'k')

for i in range(len(bsp_basis_full) - 60, len(bsp_basis_full)):
    ax.plot(r[r_th_ind:], bsp_basis_full[i, r_th_ind:], 'r')

ax.set_xlim([r[r_th_ind],r[-1]])
plt.grid(True)

#-------------------subplots tight layout--------------------------------#                    
left  = 0.01  # the left side of the subplots of the figure                                   
right = 1.0    # the right side of the subplots of the figure                                 
bottom = 0.01   # the bottom of the subplots of the figure                                    
top = 0.95      # the top of the subplots of the figure                                       
wspace = 0.2   # the amount of width reserved for blank space between subplots                
hspace = 0.2   # the amount of height reserved for white space between subplots               
plt.subplots_adjust(left=left, right=right,
                    bottom=bottom, top=top,
                    wspace=wspace, hspace=hspace)
#------------------------------------------------------------------------# 

plt.savefig('bsp_bsplines.pdf')
