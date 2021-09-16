import numpy as np
import matplotlib.pyplot as plt

t_arr = np.array([2.774, 2.078, 1.69, 1.493, 0.85, 0.77, 0.73, 0.7])
n_core_arr = np.arange(1,9)

# efficiency as compared to one core
speedup_arr = 1./(t_arr/t_arr[0])

# total time taken in days
total_time_arr = t_arr * (200 * 1500)/(3600 * 24)

fig, ax = plt.subplots(4,1,figsize=(10,12))

ax[0].plot(n_core_arr, t_arr, 'o-k')
ax[0].set_xlabel('Number of CPU cores per eigenvalue problem')
ax[0].set_ylabel('Compute time in seconds')
ax[0].grid(True)

ax[1].plot(n_core_arr, speedup_arr, 'o-k')
ax[1].set_xlabel('Number of CPU cores per eigenvalue problem')
ax[1].set_ylabel('Speedup as compared to 1 core')
ax[1].grid(True)

ax[2].plot(n_core_arr, speedup_arr/n_core_arr, 'o-k')
ax[2].set_xlabel('Number of CPU cores per eigenva\
lue problem')
ax[2].set_ylabel('Efficiency = Speedup/Cores')
ax[2].grid(True)

ax[3].plot(n_core_arr, total_time_arr, 'o-k')
ax[3].set_ylim([0,10])
ax[3].set_xlabel('Number of CPU cores per eigenvalue problem')
ax[3].set_ylabel('Total time in days')
ax[3].grid(True)

plt.tight_layout()

plt.savefig('Speedup_CPU.pdf')
