import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rescale(a):
    return (a - a.mean())/(a.max() - a.min())

def get_slope(y, x):
    return (y[1:].values - y[:-1].values)/(x[1:].values - x[:-1].values)

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"

reg_data = pd.read_csv(f"{package_dir}/reg_misfit.txt")

data_mf_rescaled = rescale(reg_data['data-misfit'])
model_mf_rescaled = rescale(reg_data['model-misfit'])
slope = get_slope(model_mf_rescaled, data_mf_rescaled)
knee_idx = np.argmin(abs(slope+1))

range_x = reg_data['data-misfit'].max() - reg_data['data-misfit'].min()
range_y = reg_data['model-misfit'].max() - reg_data['model-misfit'].min()


plt.figure()
plt.plot(reg_data['data-misfit'], 
         reg_data['model-misfit'], '+k')
plt.plot(reg_data['data-misfit'][knee_idx], 
         reg_data['model-misfit'][knee_idx], marker='s', color='red')
plt.text(reg_data['data-misfit'][knee_idx]+0.04*range_x, 
         reg_data['model-misfit'][knee_idx]+0.04*range_y,
         f"$\\mu$ = {reg_data['mu'][knee_idx]:.5f}")
plt.xlabel('Data misfit')
plt.ylabel('Model misfit')
plt.title('L-curve')
plt.show()
