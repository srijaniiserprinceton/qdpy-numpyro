from scipy.interpolate import splev
import numpy as np

def get_wsr_from_spline(r_spline, ctrl_arr,
                        knot_arr, spl_deg):
    bsp_basis = np.load('bsp_basis_full.npy')
    wsr_1 = ctrl_arr[0] @ bsp_basis
    wsr_3 = ctrl_arr[1] @ bsp_basis
    wsr_5 = ctrl_arr[2] @ bsp_basis
    
    return np.array([wsr_1, wsr_3, wsr_5])
