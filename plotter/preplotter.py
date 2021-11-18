import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import gen_wsr

plt.rcParams['axes.grid'] = True

class preplotter:
    def __init__(self, r, OM, wsr_dpt, wsr_fixed,
                 ctrl_arr_up, ctrl_arr_lo,
                 ctrl_arr_dpt_full, ctrl_arr_dpt_clipped,
                 t_internal, knot_ind_th, spl_deg=3):
    
        self.r = r
        self.OM = OM
        self.wsr_dpt = wsr_dpt
        self.wsr_fixed = wsr_fixed 
        self.ctrl_arr_up = ctrl_arr_up
        self.ctrl_arr_lo = ctrl_arr_lo
        self.ctrl_arr_dpt_full = ctrl_arr_dpt_full
        self.ctrl_arr_dpt_clipped = ctrl_arr_dpt_clipped
        self.t_internal = t_internal
        self.knot_ind_th = knot_ind_th
        self.spl_deg = spl_deg

        # plotting
        self.plot_wsr_spline_accuracy()
        self.plot_wsr_extreme()

    def plot_wsr_spline_accuracy(self):
        
        fig, ax = plt.subplots(3, 2, figsize=(15,7), sharex=True)

        # plot the wsr from dpt (no-spline)
        ax[0,0].plot(self.r, self.wsr_dpt[0], 'k')
        ax[1,0].plot(self.r, self.wsr_dpt[1], 'k')
        ax[2,0].plot(self.r, self.wsr_dpt[2], 'k')
        
        '''
        # constructing f_filtered to cross-check
        knot_num = 100
        r_spacing = len(self.r)//knot_num
        r_filtered = self.r[::r_spacing]
        '''
        
        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        '''
        wsr_spl_full = gen_wsr.get_wsr_from_spline(r_filtered, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)
        '''

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        
        # overplotting the reconstructed profile
        ax[0,0].plot(self.r, wsr_spl_full[0], '--r', alpha=0.5)
        ax[1,0].plot(self.r, wsr_spl_full[1], '--r', alpha=0.5)
        ax[2,0].plot(self.r, wsr_spl_full[2], '--r', alpha=0.5)
        
        '''
        ax[0].plot(r_filtered, wsr_spl_full[0], '--r', alpha=0.5)
        ax[1].plot(r_filtered, wsr_spl_full[1], '--r', alpha=0.5)
        ax[2].plot(r_filtered, wsr_spl_full[2], '--r', alpha=0.5)
        

        # settin axis labels
        ax[2].set_xlabel('$r$ in $R_{\odot}$', size=16)
        ax[0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)
        '''
        
        # settin axis labels
        ax[2,0].set_xlabel('$r$ in $R_{\odot}$', size=16)
        ax[0,0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1,0].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2,0].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)


        # ax[2].set_xlim([0, 1])
        
        ax[0,0].set_title('Testing spline accuracy', size=16)
        
        # plotting the error percentages
        w1r_errperc = self.get_percent_error(wsr_spl_full[0], self.wsr_dpt[0])
        w3r_errperc = self.get_percent_error(wsr_spl_full[1], self.wsr_dpt[1])
        w5r_errperc = self.get_percent_error(wsr_spl_full[2], self.wsr_dpt[2])
        
        ax[0,1].plot(self.r, w1r_errperc, 'r', alpha=0.5)
        ax[1,1].plot(self.r, w3r_errperc, 'r', alpha=0.5)
        ax[2,1].plot(self.r, w5r_errperc, 'r', alpha=0.5)
        
        # settin axis labels
        ax[2,1].set_xlabel('$r$ in $R_{\odot}$', size=16)
        ax[0,1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1,1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2,1].set_ylabel('% offset in $w_5(r)$', size=14)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig('wsr_splined.pdf')

    def plot_wsr_extreme(self):
        # getting the dpt profile from clipped dpt
        ctrl_arr_dpt_recon = np.zeros_like(self.ctrl_arr_dpt_full)
        ctrl_arr_dpt_recon[:, self.knot_ind_th:] = self.ctrl_arr_dpt_clipped
        wsr_dpt_recon = gen_wsr.get_wsr_from_spline(self.r, ctrl_arr_dpt_recon,
                                                    self.t_internal, self.spl_deg)

        # getting the upex profile
        ctrl_arr_up_full = np.zeros_like(self.ctrl_arr_dpt_full)
        ctrl_arr_up_full[:, self.knot_ind_th:] = self.ctrl_arr_up
        wsr_up = gen_wsr.get_wsr_from_spline(self.r, ctrl_arr_up_full,
                                             self.t_internal, self.spl_deg)
        
        # getting the loex profile
        ctrl_arr_lo_full = np.zeros_like(self.ctrl_arr_dpt_full)
        ctrl_arr_lo_full[:, self.knot_ind_th:] = self.ctrl_arr_lo
        wsr_lo = gen_wsr.get_wsr_from_spline(self.r, ctrl_arr_lo_full,
                                             self.t_internal, self.spl_deg)
        
        # plotting 
        fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex = True)
        
        # the dpt reconstructed from spline
        ax[0].plot(self.r, self.wsr_fixed[0] + wsr_dpt_recon[0], 'k')
        ax[1].plot(self.r, self.wsr_fixed[1] + wsr_dpt_recon[1], 'k')
        ax[2].plot(self.r, self.wsr_fixed[2] + wsr_dpt_recon[2], 'k')

        ax[0].plot(self.r, self.wsr_dpt[0], '--r')
        ax[1].plot(self.r, self.wsr_dpt[1], '--r')
        ax[2].plot(self.r, self.wsr_dpt[2], '--r')


        # saving the profile being used in 1walk1iter_sparse.py forusing in qdpt.py
        wsr_pyro = self.wsr_fixed + wsr_dpt_recon
        np.save('wsr_pyro.npy', wsr_pyro)
        
        # shading the area where it is allowed to vary
        ax[0].fill_between(self.r, self.wsr_fixed[0] + wsr_up[0],
                           self.wsr_fixed[0] + wsr_lo[0],
                           color='gray', alpha=0.5)
        
        ax[1].fill_between(self.r, self.wsr_fixed[1] + wsr_up[1],
                           self.wsr_fixed[1] + wsr_lo[1],
                           color='gray', alpha=0.5)
        
        ax[2].fill_between(self.r, self.wsr_fixed[2] + wsr_up[2],
                           self.wsr_fixed[2] + wsr_lo[2],
                           color='gray', alpha=0.5)

        plt.tight_layout()

        plt.savefig('wsr_extreme.pdf')

    def get_percent_error(self, a1, a2):
        errperc = np.zeros_like(a1)
        diff = a1 - a2
        mask0 = abs(a1) < 1e-6
        errperc[~mask0] = abs(diff)[~mask0]*100/abs(a1)[~mask0]
        return errperc

    
