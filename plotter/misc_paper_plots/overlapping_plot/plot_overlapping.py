import matplotlib.pyplot as plt
from scipy.special import lpmn
import os
import numpy as np
import pandas as pd

# directory in which the output is stored                                                     
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]


# GVARS.OM                                                                                    
GVARS_OM = np.load('GVARS_OM.npy')
# radial grid                                                                                 
r = np.load('r.npy')

s = np.array([1, 3, 5])
scale_fac = np.sqrt((2*s + 1)/4./np.pi)
unitconv = GVARS_OM * 1e9

# theta grid                                                                                
theta = np.linspace(0.0, np.pi, 180)
legpoly = np.zeros((len(theta), len(s)))
for i in range(len(theta)):
    th = theta[i]
    legpoly[i] = lpmn(0, 5, np.cos(th))[1][0, 1::2]


# computing Omega(r, theta)                                                                  
def get_Omega_r_theta(wsr):
    Omega_r_theta = (legpoly*scale_fac) @ wsr / r * unitconv
    return Omega_r_theta

fig = plt.figure(figsize=(15, 7))

# the list of axis
ax_list = []

# loading the daylist                                                                         
pt = pd.read_table(f'{scratch_dir}/input_files/daylist.txt', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

dayind_solmin = 53
day_solmin = pt['MDI'][dayind_solmin]
wsr_dpt_solmin = np.load(f'wsr_dpy_fit_{day_solmin}.npy')
Omega_r_theta_dpt_solmin = get_Omega_r_theta(wsr_dpt_solmin)

# making the axes
for i in range(31,60):
    ax_i = fig.add_axes([-0.2 + 0.06 * i, 0.1, 0.85, 0.85], zorder=9-i)
    ax_list.append(ax_i)
    ax_i.set_aspect('equal')
    ax_i.patch.set_alpha(0.0)

rr, tt = np.meshgrid(r, -np.pi/2. + theta)
xx, yy = rr * np.cos(tt), rr * np.sin(tt)

# plotting in the axes
for dayind in range(31,31+10):    
    day = pt['MDI'][dayind]
    try:
        wsr_dpt = np.load(f'wsr_dpy_fit_{day}.npy')
    except:
        print(f'Bad day: {day}; skipping')
        continue

    Omega_r_theta_dpt = get_Omega_r_theta(wsr_dpt)
    
    ax_list[i-31].pcolormesh(xx, yy, Omega_r_theta_dpt_solmin - Omega_r_theta_dpt_solmin,
                          cmap='jet_r', rasterized=True,
                          vmin=-5, vmax=5)
    
    
fig.savefig('overlaping_plot.pdf')
