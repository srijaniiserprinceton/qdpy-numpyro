import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

a2data = pd.read_csv('ella2.csv')
a4data = pd.read_csv('ella4.csv')

nu_a2 = a2data['nu']
nu_a4 = a4data['nu']
ella2 = a2data['ella2']
ella4 = a4data['ella4']

plt.figure()
plt.plot(nu_a2, ella2, 'xk')
plt.plot(nu_a4, ella4, 'xr')
plt.show()

hmi_data_dir = "/mnt/disk2/samarth/qdpy-numpyro/qdpy_iofiles/input_files/hmi"
hmi_data = np.loadtxt(f"{hmi_data_dir}/hmi.in.72d.6328.36")
ell_data = hmi_data[:, 0]
enn_data = hmi_data[:, 1]
nu_data  = hmi_data[:, 2]
mask0 = enn_data == 0
mask1 = enn_data == 1

ell0_data = ell_data[mask0]
nu0_data = nu_data[mask0] * 1e-3

a2interp = interp1d(nu_a2, ella2, kind="linear", fill_value="extrapolate")
a4interp = interp1d(nu_a4, ella4, kind="linear", fill_value="extrapolate")

ella2_obs = a2interp(nu0_data)
ella4_obs = a4interp(nu0_data)

a2_obs = ella2_obs#/ell0_data
a4_obs = ella4_obs#/ell0_data

plt.figure()
plt.plot(ell0_data, a2_obs, '+k')
plt.plot(ell0_data, a4_obs, 'xr')
plt.show()



