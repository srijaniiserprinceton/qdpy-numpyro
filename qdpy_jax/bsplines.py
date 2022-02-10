import os
os.environ["F90"] = "gfortran"
import sys
import numpy as np
from avni.tools.bases import eval_splrem, eval_polynomial

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
        self.t_internal = None
        self.c_arr_dpt_full = None
        self.c_arr_dpt_clipped = None
        self.c_arr_up = None
        self.c_arr_lo = None
        self.wsr_fixed = None
        self.bsp_basis = None
        self.d_bsp_basis = None
        self.d2_bsp_basis = None

        # performing different tasks
        self.create_knots_and_bsp()
        self.get_uplo_dpt_carr()
        self.get_fixed_wsr()

    def create_knots_and_bsp(self):
        rmin, rmax = self.r.min(), self.r.max()
        total_knot_num = int(np.round((rmax-rmin)/(1 - self.rth))) \
                         * self.custom_knot_num
        
        knot_locs = np.linspace(rmin, rmax, total_knot_num)
        
        self.knot_ind_th = np.argmin(abs(knot_locs - self.rth))
        self.knot_locs = knot_locs

        vercof1, dvercof1 = eval_polynomial(self.r, [rmin, rmax],
                                            1, types= ['TOP','BOTTOM'])
        vercof2, dvercof2 = eval_splrem(self.r, [rmin, rmax],
                                        len(knot_locs))
        
        bsp_basis = np.column_stack((vercof1, vercof2[:, 1:-1]))
        d_bsp_basis = np.column_stack((dvercof1, dvercof2[:, 1:-1]))
        
        # storing the analytically derived B-splines and it first derivatives
        # making them of shape (n_basis, r)
        self.bsp_basis = bsp_basis.T
        self.d_bsp_basis = d_bsp_basis.T

    def get_uplo_dpt_carr(self):
        # creating the carr corresponding to the DPT using custom knots
        Gtg = self.bsp_basis @ self.bsp_basis.T   # shape(n_basis, n_basis)
        c_arr_dpt = []
        for s_ind in range(self.len_s):
            c_arr_dpt.append(np.linalg.inv(Gtg) @ (self.bsp_basis @ self.wsr[s_ind]))
        
        self.c_arr_dpt_full = np.array(c_arr_dpt)
        
        # clipping it off below rth (approximately)
        self.c_arr_dpt_clipped = self.c_arr_dpt_full[:, self.knot_ind_th:]
        
        # creating the upex and loex controls
        c_arr_up = self.fac_arr[0][:, np.newaxis] * self.c_arr_dpt_clipped
        c_arr_lo = self.fac_arr[1][:, np.newaxis] * self.c_arr_dpt_clipped
        
        # swapping according to which ones are larger
        swap_mask = c_arr_up < c_arr_lo

        c_arr_up_temp = c_arr_up.copy()
        c_arr_up[swap_mask] = c_arr_lo[swap_mask]
        c_arr_lo[swap_mask] = c_arr_up_temp[swap_mask]

        self.c_arr_up = c_arr_up
        self.c_arr_lo = c_arr_lo

    def get_fixed_wsr(self):
        # creating the fixed control points
        c_arr_fixed = np.zeros_like(self.c_arr_dpt_full)
        c_arr_fixed[:, :self.knot_ind_th] = self.c_arr_dpt_full[:, :self.knot_ind_th]
        wsr_fixed = []
        
        for i in range(self.len_s):
            wsr_fixed.append(c_arr_fixed[i, :] @ self.bsp_basis)

        self.wsr_fixed = np.array(wsr_fixed)
