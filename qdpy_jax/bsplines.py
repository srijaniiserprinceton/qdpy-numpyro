import numpy as np
from scipy.interpolate import splrep, splev

class get_splines:
    def __init__(self, r, rth, wsr, custom_knot_num, fac_arr, spl_deg=3):
        self.r = r
        self.rth = rth
        self.wsr = wsr
        self.len_s = wsr.shape[0]
        self.custom_knot_num = custom_knot_num
        self.fac_arr = fac_arr
        self.spl_deg = spl_deg
        
        self.knot_ind_th = None
        self.t_scipy = None
        self.t_internal = None
        self.c_arr_full = None
        self.c_arr_dpt_full = None
        self.c_arr_dpt_clipped = None
        self.c_arr_up = None
        self.c_arr_lo = None
        self.wsr_fixed = None

        # performing different tasks
        self.create_custom_knot_arr()
        self.get_fixed_wsr()
        self.get_uplo_dpt_carr()

    def create_custom_knot_arr(self):
        # decomposing the wsr into splines
        self.t_scipy, c_1, __ = splrep(self.r, self.wsr[0], k=self.spl_deg)

        self.knot_ind_th = np.argmin(np.abs(self.t_scipy - self.rth))
        
        # the full control vector (s x nc_full)
        self.c_arr_full = c_1
        
        for s_ind in range(1, self.len_s):
            __, c, __ = splrep(self.r, self.wsr[1])
            self.c_arr = np.vstack((self.c_arr_full, c))
        
    def get_fixed_wsr(self):
        # creating the fixed control points
        c_arr_fixed = np.zeros_like(self.c_arr)
        c_arr_fixed[:self.knot_ind_th] = self.c_arr[:self.knot_ind_th]

        self.wsr_fixed = splev(self.r, (self.t_scipy, c_arr_fixed[0], self.spl_deg))
        
        # creating the fixed wsr
        for s_ind in range(self.len_s):
            self.wsr_fixed = np.vstack((splev(self.r,
                                              (self.t_scipy,
                                               c_arr_fixed,
                                               self.spl_deg)),
                                        self.wsr_fixed))

    def get_uplo_dpt_carr(self):
        # making internal knots (excluding the first external point)
        t_internal = self.t_scipy[self.spl_deg+1:self.knot_ind_th]
        # putting certain number of knots in the surface part (excluding the outermost point)
        t_internal = np.append(t_internal, np.linspace(self.t_scipy[self.knot_ind_th],
                                                       self.t_scipy[-(self.spl_deg+2)],
                                                       self.custom_knot_num))
        self.t_internal = t_internal
        
        # creating the carr corresponding to the DPT using custom knots
        self.t_scipy, c_1, __ = splrep(self.r,
                                       self.wsr[0],
                                       k=self.spl_deg,
                                       t=t_internal)
        c_arr_dpt = c_1
        
        # storing all other s
        for s_ind in range(1, self.len_s):
            __, c, __ = splrep(self.r,
                               self.wsr[1],
                               k=self.spl_deg,
                               t=t_internal)
            c_arr_dpt = np.vstack((c_arr_dpt, c))

        
        self.c_arr_dpt_full = c_arr_dpt
        
        # clipping it off below rth (approximately)
        self.c_arr_dpt_clipped = c_arr_dpt[:, self.knot_ind_th:]
        
        # creating the upex and loex controls
        c_arr_up = self.fac_arr[0][:,np.newaxis] * self.c_arr_dpt_clipped
        c_arr_lo = self.fac_arr[1][:,np.newaxis] * self.c_arr_dpt_clipped

        # swapping according to which ones are larger
        swap_mask = c_arr_up < c_arr_lo

        c_arr_up_temp = c_arr_up
        c_arr_up[swap_mask] = c_arr_lo[swap_mask]
        c_arr_lo[swap_mask] = c_arr_up_temp[swap_mask]

        self.c_arr_up = c_arr_up
        self.c_arr_lo = c_arr_lo
        
