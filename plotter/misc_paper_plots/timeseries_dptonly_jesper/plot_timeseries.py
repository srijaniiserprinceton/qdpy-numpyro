import os
import numpy as np
import pandas as pd
from scipy.special import lpmn
import matplotlib.pyplot as plt
plt.ion()

# directory in which the output is stored                                                     
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

# GVARS.OM 
GVARS_OM = np.load('GVARS_OM.npy')

# loading the daylist 
pt = pd.read_table(f'{scratch_dir}/input_files/daylist.txt', delim_whitespace=True,
                   names=('SN', 'MDI', 'DATE'),
                   dtype={'SN': np.int64,
                          'MDI': np.int64,
                          'DATE': str})

dayind_min, dayind_max = 0, 59

baddays = np.array([10432, 9352, 10072, 9712, 10288])

s = np.array([1, 3, 5])
scale_fac = np.sqrt((2*s + 1)/4./np.pi)
unitconv = GVARS_OM * 1e9

# radial grid
r = np.load('r.npy')

# theta grid
theta = np.linspace(0.0, np.pi, 180)
legpoly = np.zeros((len(theta), len(s)))
for i in range(len(theta)):
    th = theta[i]
    legpoly[i] = lpmn(0, 5, np.cos(th))[1][0, 1::2]

rplot = 0.98
rplot_ind = np.argmin(np.abs(r - rplot))

# computing Omega(r, theta)
def get_Omega_r_theta(wsr):    
    Omega_r_theta = (legpoly*scale_fac) @ wsr / r * unitconv
    
    return Omega_r_theta


# choosing the solar minima with respect to which we will plot
# https://www.weather.gov/news/201509-solar-cycle says it happened in Dec, 2019
dayind_solmin = 3
day_solmin = pt['MDI'][dayind_solmin]
wsr_dpt_solmin = np.load(f'wsr_dpy_fit_{day_solmin}.npy')

# array to store the wsr_dpt timeseries at rplot
Omega_r_theta_dpt_solmin = get_Omega_r_theta(wsr_dpt_solmin)
Omega_dpt_timearr = np.array([Omega_r_theta_dpt_solmin[:, rplot_ind]])

Omega_hybrid_timearr = Omega_dpt_timearr * 1.0

# list of days
dayarr = []

for dayind in range(dayind_min, dayind_max+1):
    day =  pt['MDI'][dayind]
    dayarr.append(day)
    print(day)

    try:
        wsr_dpt = np.load(f'wsr_dpy_fit_{day}.npy')
    except:
        print(f'Bad day: {day}; skipping')
        nan_vals = np.zeros((1,len(theta))) + np.nan
        Omega_dpt_timearr = np.append(Omega_dpt_timearr, nan_vals, axis=0)
        Omega_hybrid_timearr = np.append(Omega_hybrid_timearr, nan_vals, axis=0)
        continue

    Omega_r_theta_dpt = get_Omega_r_theta(wsr_dpt)
    Omega_dpt_timearr = np.append(Omega_dpt_timearr,
                                  np.array([Omega_r_theta_dpt[:, rplot_ind]]), axis=0)
    

# rotation profiles relative to the solar minina (stored in the first index)
Omega_dpt_timearr_rel = Omega_dpt_timearr[1:] - Omega_dpt_timearr[0]

# plotting the Omega(r, theta) in for the solmin profile
rr, tt = np.meshgrid(r, -np.pi/2. + theta)
xx, yy = rr * np.cos(tt), rr * np.sin(tt)

fig, ax = plt.subplots(1, 1, figsize=(5,10))
im = plt.pcolormesh(xx, yy, Omega_r_theta_dpt_solmin, cmap='jet_r', rasterized=True)
fig.colorbar(im)
ax.set_aspect('equal')
plt.savefig('Omega_r_theta_solmin.pdf')

# plotting the timeseries
dayarr = np.asarray(dayarr)
relyeararr = (dayarr - dayarr[0])/365.

yy, tt = np.meshgrid(relyeararr, -np.pi/2. + theta, indexing='ij')

plt.figure()
plt.pcolormesh(yy, tt * 180./np.pi, Omega_dpt_timearr_rel, cmap='jet',
               vmin=-4, vmax=4)
plt.colorbar()
plt.savefig('Omega_dpt_timearr_rel.pdf')
