import os
import numpy as np

def get_wsr(fname):
    c1 = np.load(fname)
    c1full = GVARS.ctrl_arr_dpt_full * 1.0
    c1full[2, GVARS.knot_ind_th:] = c1
    wsr = c1full @ GVARS.bsp_basis_full
    return c1full, wsr


def compute_misfit(arr1, arr2):
    return np.sqrt(sum(abs(arr1 - arr2)**2))

outdir = '/mnt/disk2/samarth/qdpy-numpyro/qdpy_iofiles/dpy_jax'
mu_limits = [1e-15, 1e-3]
print(f"Running python scripts")
os.system(f"cd ../dpy_jax; python run_reduced_problem_newton.py --mu {mu_limits[0]} >temp.out 2>temp.err")
os.system(f"cd ../dpy_jax; python run_reduced_problem_newton.py --mu {mu_limits[1]} >temp.out 2>temp.err")
val0 = np.load(f'{outdir}/c_arr_dpt.npy')
val1 = np.load(f'{outdir}/carr_fit_{mu_limits[0]:.5e}.npy')
val2 = np.load(f'{outdir}/carr_fit_{mu_limits[1]:.5e}.npy')

mf1 = compute_misfit(val0, val1)
mf2 = compute_misfit(val0, val2)
print(f"muval1 = {mu_limits[0]:.5e}; misfit = {mf1:.5e}")
print(f"muval1 = {mu_limits[1]:.5e}; misfit = {mf2:.5e}")

if mf1 > mf2:
    leftmu = mu_limits[0]
    rightmu = mu_limits[1]
else:
    leftval = mu_limits[1]
    rightval = mu_limits[0]

maxiter = 20

for i in range(maxiter):
    log_muval = 0.5*(np.log10(leftmu) + np.log10(rightmu))
    muval = 10**log_muval
    os.system(f"python ../dpy_jax/run_reduced_problem_newton.py --mu {muval} >temp.out 2>temp.err")

    muval1 = 10**(0.5*(np.log10(leftmu) + log_muval))
    muval2 = 10**(0.5*(np.log10(rightmu) + log_muval))
    os.system(f"cd ../dpy_jax; python run_reduced_problem_newton.py --mu {muval1} >temp.out 2>temp.err")
    os.system(f"cd ../dpy_jax; python run_reduced_problem_newton.py --mu {muval2} >temp.out 2>temp.err")
    val1 = np.load(f'{outdir}/carr_fit_{muval1:.5e}.npy')
    val2 = np.load(f'{outdir}/carr_fit_{muval2:.5e}.npy')
    mf1 = compute_misfit(val0, val1)
    mf2 = compute_misfit(val0, val2)
    print(f"muval1 = {muval1:.5e}; misfit = {mf1:.5e}")

    if mf1 < mf2:
        leftmu = muval1 * 1.0
        rightmu = muval * 1.0
    else:
        leftmu = muval * 1.0
        rightmu = muval2 * 1.0
    
