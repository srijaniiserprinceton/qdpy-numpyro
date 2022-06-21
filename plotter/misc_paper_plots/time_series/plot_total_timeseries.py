import os
import numpy as np
import pandas as pd
from scipy.special import lpmn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.ion()

# directory in which the output is stored                                                     
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]

INSTR_MDI = "mdi"
INSTR_HMI = "hmi"

orgfiles_dir_mdi = f"{scratch_dir}/{INSTR_MDI}-run1/organized-files"
orgfiles_dir_hmi = f"{scratch_dir}/{INSTR_HMI}-run1/organized-files"
plot_dir = f"{scratch_dir}/total_time_series"

GVARS_OM = np.load(f'{orgfiles_dir_mdi}/plot_files/GVARS_OM.npy')

# loading the daylists for mdi and hmi
pt_mdi = pd.read_table(f'{package_dir}/preprocess/daylist.mdi', delim_whitespace=True,
                       names=('SN', 'MDI', 'DATE'),
                       dtype={'SN': np.int64,
                              'MDI': np.int64,
                              'DATE': str})

pt_hmi = pd.read_table(f'{package_dir}/preprocess/daylist.hmi', delim_whitespace=True,
                       names=('SN', 'MDI', 'DATE'),
                       dtype={'SN': np.int64,
                              'MDI': np.int64,
                              'DATE': str})

# defining the dayind for mdi and hmi respectively.
# mdi ends at 6328 and hmi starts at 6400 and goes upto 9136
dayind_min_arr, dayind_max_arr = np.array([0, 1]), np.array([68, 39])

s = np.array([1, 3, 5, 7, 9])
scale_fac = np.sqrt((2*s + 1)/4./np.pi)
unitconv = GVARS_OM * 1e9

# radial grid
r = np.load(f'{orgfiles_dir_mdi}/plot_files/r.npy')

# theta grid
theta = np.linspace(0.0, np.pi, 180)
legpoly = np.zeros((len(theta), len(s)))
for i in range(len(theta)):
    th = theta[i]
    legpoly[i] = lpmn(0, s.max(), np.cos(th))[1][0, 1::2]

# the depth at which the latitudinal variation plot is made
rplot = 1.0
rplot_ind = np.argmin(np.abs(r - rplot))

# computing Omega(r, theta)
def get_Omega_r_theta(wsr):    
    Omega_r_theta = (legpoly*scale_fac) @ wsr / r * unitconv
    
    return -1.0 * Omega_r_theta # -1 to match the convention of Vorontsov (2002)


# choosing the solar minima with respect to which we will plot
# https://www.weather.gov/news/201509-solar-cycle says it happened in Dec, 2019
dayind_solmin = 0 # 50 for Dec, 2019
day_solmin = pt_mdi['MDI'][dayind_solmin]
wsr_dpt_solmin = np.load(f'{orgfiles_dir_mdi}/plot_files/wsr_dpy_fit_{day_solmin}.npy')
wsr_hybrid_solmin = np.load(f'{orgfiles_dir_mdi}/plot_files/wsr_hybrid_fit_{day_solmin}.npy')
# array to store the wsr_dpt timeseries at rplot
Omega_r_theta_dpt_solmin = get_Omega_r_theta(wsr_dpt_solmin)
Omega_r_theta_hybrid_solmin = get_Omega_r_theta(wsr_hybrid_solmin)


def make_timeseries_arr(dayind_min, dayind_max, pt, instr_dir):

    Omega_dpt_timearr = np.array([Omega_r_theta_dpt_solmin[:, rplot_ind]])
    Omega_hybrid_timearr = Omega_dpt_timearr * 1.0
    
    # list of days
    dayarr = []

    for dayind in range(dayind_min, dayind_max+1):
        day =  pt['MDI'][dayind]
        dayarr.append(day)
                    
        try:
            wsr_dpt = np.load(f'{instr_dir}/plot_files/wsr_dpy_fit_{day}.npy')
            wsr_hybrid = np.load(f'{instr_dir}/plot_files/wsr_hybrid_fit_{day}.npy')
        except FileNotFoundError:
            print(f"Bad day = {day}")
            nan_vals = np.zeros((1, len(theta))) + np.nan
            Omega_dpt_timearr = np.append(Omega_dpt_timearr, nan_vals, axis=0)
            Omega_hybrid_timearr = np.append(Omega_hybrid_timearr, nan_vals, axis=0)
            continue
        
        print(day)
        Omega_r_theta_dpt = get_Omega_r_theta(wsr_dpt)
        Omega_dpt_timearr = np.append(Omega_dpt_timearr,
                                      np.array([Omega_r_theta_dpt[:, rplot_ind]]),
                                      axis=0)
        
        Omega_r_theta_hybrid = get_Omega_r_theta(wsr_hybrid)
        Omega_hybrid_timearr = np.append(Omega_hybrid_timearr,
                                         np.array([Omega_r_theta_hybrid[:, rplot_ind]]),
                                         axis=0)
        
    return (dayarr, Omega_r_theta_dpt, Omega_r_theta_hybrid,
            Omega_dpt_timearr, Omega_hybrid_timearr)

# getting the arrays for mdi
dayarr_mdi, Omega_r_theta_dpt_mdi, Omega_r_theta_hybrid_mdi,\
Omega_dpt_timearr_mdi, Omega_hybrid_timearr_mdi = make_timeseries_arr(dayind_min_arr[0],
                                                                      dayind_max_arr[0],
                                                                      pt_mdi,
                                                                      f'{orgfiles_dir_mdi}')
# getting the arrays for hmi
dayarr_hmi, Omega_r_theta_dpt_hmi, Omega_r_theta_hybrid_hmi,\
Omega_dpt_timearr_hmi, Omega_hybrid_timearr_hmi = make_timeseries_arr(dayind_min_arr[1],
                                                                      dayind_max_arr[1],
                                                                      pt_hmi,
                                                                      f'{orgfiles_dir_hmi}')

# concatenating mdi with hmi
dayarr = np.append(dayarr_mdi, dayarr_hmi)
Omega_r_theta_dpt = np.append(Omega_r_theta_dpt_mdi,
                              Omega_r_theta_dpt_hmi[1:], axis=0)
Omega_r_theta_hybrid = np.append(Omega_r_theta_hybrid_mdi,
                                 Omega_r_theta_hybrid_hmi[1:], axis=0)
Omega_dpt_timearr = np.append(Omega_dpt_timearr_mdi,
                              Omega_dpt_timearr_hmi[1:], axis=0)
Omega_hybrid_timearr = np.append(Omega_hybrid_timearr_mdi,
                                 Omega_hybrid_timearr_hmi[1:], axis=0)

# rotation profiles relative to the solar minina (stored in the first index)
Omega_dpt_timearr_rel = Omega_dpt_timearr[1:] - Omega_dpt_timearr[0]

# plotting the Omega(r, theta) in for the solmin profile
rr, tt = np.meshgrid(r, -np.pi/2. + theta)
xx, yy = rr * np.cos(tt), rr * np.sin(tt)


fig, ax = plt.subplots(1, 1, figsize=(5, 10))
im = plt.pcolormesh(xx, yy, Omega_r_theta_dpt_solmin, cmap='jet_r', rasterized=True)
fig.colorbar(im)
ax.set_aspect('equal')
plt.savefig(f'{plot_dir}/Omega_r_theta_solmin.pdf')

fig, ax = plt.subplots(1, 1, figsize=(5, 10))
meshval = Omega_r_theta_dpt_solmin - Omega_r_theta_hybrid_solmin
im = plt.pcolormesh(xx, yy, meshval, cmap='jet_r', rasterized=True)
fig.colorbar(im)
ax.set_aspect('equal')
plt.savefig(f'{plot_dir}/Omega_r_theta_solmin-diff.pdf')


# plotting the timeseries
Omega_hybrid_timearr_rel = Omega_hybrid_timearr[1:] - Omega_hybrid_timearr[0]

# plotting the timeseries
dayarr = np.asarray(dayarr)
relyeararr = (dayarr - dayarr[0])/365.

mdi_hmi_switch_idx = np.argmin(np.abs(dayarr - 6328))

yy, tt = np.meshgrid(relyeararr, -np.pi/2. + theta, indexing='ij')
plt.figure(figsize=(30,10))
meshval = Omega_dpt_timearr_rel
masknan = ~np.isnan(meshval)
meshmax = abs(meshval[masknan]).max()
meshmax = 4. # hardcoding
plt.pcolormesh(yy, tt * 180./np.pi, meshval, cmap='jet', vmin=-meshmax, vmax=meshmax)
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{plot_dir}/Omega_dpt_timearr_rel.pdf')

plt.figure(figsize=(30,10))
meshval = Omega_hybrid_timearr_rel
masknan = ~np.isnan(meshval)
meshmax = abs(meshval[masknan]).max()
plt.pcolormesh(yy, tt * 180./np.pi, meshval, vmin=-meshmax, vmax=meshmax, cmap='jet')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{plot_dir}/Omega_hybrid_timearr_rel.pdf')


plt.figure(figsize=(30,10))
meshval = Omega_hybrid_timearr[1:] - Omega_dpt_timearr[1:]
masknan = ~np.isnan(meshval)
meshmax = abs(meshval[masknan]).max()
meshmax = 0.0065
plt.pcolormesh(yy, tt * 180./np.pi, meshval, vmin=-meshmax, vmax=meshmax, cmap='jet')
plt.axvline(relyeararr[mdi_hmi_switch_idx]+(5./48.), linestyle='--', color='w',
            lw = 3)
my_xticks = ['John','Arnold','Mavis','Matt']
plt.xticks(relyeararr[::15], (relyeararr[::15] + 1996.4).astype('int'), rotation=45)
plt.ylabel('Latitude $\lambda$ in degrees')
plt.xlabel('Years')
plt.title('$\Omega_{\mathrm{hyb}}(R_{\odot}, \lambda)' +
          '- \Omega_{\mathrm{DPT}}(R_{\odot}, \lambda)$ in nHz')
plt.colorbar(pad=0.01)
plt.tight_layout()
plt.savefig(f'{plot_dir}/Omega_hybrid_rel_dpt.pdf',bbox_inches='tight')
