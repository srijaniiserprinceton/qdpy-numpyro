from scipy.interpolate import splev
import numpy as np

def get_wsr_from_spline(r_spline, ctrl_arr,
                        knot_arr, spl_deg=3):
    wsr_1 = splev(r_spline, (knot_arr,
                  ctrl_arr[0], spl_deg))
    wsr_3 = splev(r_spline, (knot_arr,
                  ctrl_arr[1], spl_deg))
    wsr_5 = splev(r_spline, (knot_arr,
                  ctrl_arr[2], spl_deg))
    
    return np.array([wsr_1, wsr_3, wsr_5])
