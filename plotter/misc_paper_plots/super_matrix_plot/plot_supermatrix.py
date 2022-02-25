import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import numpy as np

# loading the qdpy_jax supermatrix in muHz
supmat = np.load('supmat_qdpt_200.npy')

# loading the qdpy cenmult neighbours
cnm_nbs = np.load('CNM_NBS_200.npy')

#--------------Making locations for axhline and axvline-------------#
two_lp1_cumulative = np.cumsum(2 * cnm_nbs[:,1] + 1)

#--------------------Plotting-----------------------#

fig, ax = plt.subplots(1, 1, figsize=(12,10), facecolor='w', edgecolor='k')

# prepare x and y for scatter plot
plot_list = []
for rows,cols in zip(np.where(supmat!=0)[0],np.where(supmat!=0)[1]):
    plot_list.append([cols,rows,supmat[rows,cols]])
plot_list = np.array(plot_list)

# scatter plot with color bar, with rows on y axis
im = ax.scatter(plot_list[:,0],plot_list[:,1],c=plot_list[:,2], s=5,
                norm = colors.SymLogNorm(linthresh=1e-1,
                                         linscale=1e-1,
                                         vmin = np.min(supmat),
                                         vmax = np.max(supmat)),
                cmap ='seismic')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1%", pad=0.5)
   
cb = plt.colorbar(im, cax=cax)

# tick_locs = np.append(-np.logspace(2,1,3), np.append([0], np.logspace(1,2,3)))
tick_locs = np.array([-100, -10, -1, 0, 1, 10, 100])
cb.set_ticks(tick_locs)

cb.set_ticklabels(["$-100$", "$-1$", "$-0.1$", "$0$", "$0.1$", "$1$", "$100$"])

cb.ax.tick_params(labelsize=16)

for axline in two_lp1_cumulative[:-1]:
    ax.axhline(y=axline, color='gray', alpha=0.5)
    ax.axvline(x=axline, color='gray', alpha=0.5)

# full range for x and y axes
ax.set_xlim(0,supmat.shape[1])
ax.set_ylim(0,supmat.shape[0])

# turning off ticks
ax.set_xticks([], [])
ax.set_yticks([], [])

# putting the text for mode labels
x_offset = 0.0
for i, ell0 in enumerate(cnm_nbs[:,1]):
    x_loc = ell0 + x_offset
    ax.text(x_loc - 50, 0 - 50, '${}_{0}S_{%i}$'%ell0, fontsize=20)
    x_offset = two_lp1_cumulative[i]
    

y_offset = 0.0
for i, ell0 in enumerate(cnm_nbs[:,1]):
    y_loc = ell0 + y_offset
    ax.text(0 - 180, y_loc + 0, '${}_{0}S_{%i}$'%ell0, fontsize=20)
    y_offset = two_lp1_cumulative[i]


# plt.gca().invert_xaxis()
ax.invert_yaxis()

ax.set_aspect('equal')

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

plt.savefig('supmat_paper.pdf')


