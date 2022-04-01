from scipy.interpolate import splev
import numpy as np

def get_wsr_from_spline(GVARS, r_spline, ctrl_arr,
                        knot_arr, spl_deg):
    bsp_basis = GVARS.bsp_basis_full
    num_s = ctrl_arr.shape[0]
    wsr = []
    for i in range(num_s):
        wsr.append(ctrl_arr[i] @ bsp_basis)
    return np.array(wsr)
