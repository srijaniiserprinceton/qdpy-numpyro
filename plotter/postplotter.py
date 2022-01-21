import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmn
from tqdm import tqdm

from qdpy_jax import gen_wsr

plt.rcParams['axes.grid'] = True

class postplotter:
    def __init__(self, GVARS, ctrl_arr_fit_full, tag):
        self.GVARS = GVARS
        self.r = GVARS.r
        self.OM = GVARS.OM
        self.wsr_dpt = GVARS.wsr
        self.ctrl_arr_dpt_full = ctrl_arr_fit_full
        self.t_internal = GVARS.t_internal
        self.knot_ind_th = GVARS.knot_ind_th
        self.spl_deg = GVARS.spl_deg
        self.tag = tag

        # plotting
        self.plot_fit_wsr()

    def plot_fit_wsr(self):
        fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)

        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r, self.wsr_dpt[0], 'k')
        ax[1, 0].plot(self.r, self.wsr_dpt[1], 'k')
        ax[2, 0].plot(self.r, self.wsr_dpt[2], 'k')

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        # overplotting the reconstructed profile
        ax[0, 0].plot(self.r, wsr_spl_full[0], '--r', alpha=0.5)
        ax[1, 0].plot(self.r, wsr_spl_full[1], '--r', alpha=0.5)
        ax[2, 0].plot(self.r, wsr_spl_full[2], '--r', alpha=0.5)

        # settin axis labels and title
        ax[0, 0].set_title(f'$w_s(r)$ DPT vs. {self.tag}', size=16)
        ax[0, 0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1, 0].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_xlabel('$r$ in $R_{\odot}$', size=16)
        
        # plotting the error percentages
        w1r_errperc = self.get_percent_error(wsr_spl_full[0], self.wsr_dpt[0])
        w3r_errperc = self.get_percent_error(wsr_spl_full[1], self.wsr_dpt[1])
        w5r_errperc = self.get_percent_error(wsr_spl_full[2], self.wsr_dpt[2])
        
        ax[0, 1].semilogy(self.r, abs(w1r_errperc), 'r', alpha=0.5)
        ax[1, 1].semilogy(self.r, abs(w3r_errperc), 'r', alpha=0.5)
        ax[2, 1].semilogy(self.r, abs(w5r_errperc), 'r', alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[2, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.tag}_wsr.pdf')
        plt.close()

    def plot_fit_wsr_zoom(self):
        fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)

        rth_idx = np.argmin(abs(self.r - self.GVARS.rth))

        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r[rth_idx:], self.wsr_dpt[0][rth_idx:], 'k')
        ax[1, 0].plot(self.r[rth_idx:], self.wsr_dpt[1][rth_idx:], 'k')
        ax[2, 0].plot(self.r[rth_idx:], self.wsr_dpt[2][rth_idx:], 'k')

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        # overplotting the reconstructed profile
        ax[0, 0].plot(self.r[rth_idx:], wsr_spl_full[0][rth_idx:], '--r', alpha=0.5)
        ax[1, 0].plot(self.r[rth_idx:], wsr_spl_full[1][rth_idx:], '--r', alpha=0.5)
        ax[2, 0].plot(self.r[rth_idx:], wsr_spl_full[2][rth_idx:], '--r', alpha=0.5)

        # settin axis labels and title
        ax[0, 0].set_title(f'$w_s(r)$ DPT vs. {self.tag}', size=16)
        ax[0, 0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1, 0].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_xlabel('$r$ in $R_{\odot}$', size=16)
        
        # plotting the error percentages
        w1r_errperc = self.get_percent_error(wsr_spl_full[0], self.wsr_dpt[0])
        w3r_errperc = self.get_percent_error(wsr_spl_full[1], self.wsr_dpt[1])
        w5r_errperc = self.get_percent_error(wsr_spl_full[2], self.wsr_dpt[2])
        
        ax[0, 1].semilogy(self.r[rth_idx:], abs(w1r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[1, 1].semilogy(self.r[rth_idx:], abs(w3r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[2, 1].semilogy(self.r[rth_idx:], abs(w5r_errperc)[rth_idx:], 'r', alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[2, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.tag}_wsr_zoom.pdf')
        plt.close()

    def plot_omega_rtheta(self, theta=np.arange(0, 90, 15)):
        fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)

        # rth_idx = np.argmin(abs(self.r - self.GVARS.rth - 0.1))
        rth_idx = np.argmin(abs(self.r - 0.5))
        rlist = self.r[rth_idx:]
        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r[rth_idx:], self.wsr_dpt[0][rth_idx:], 'k')
        ax[1, 0].plot(self.r[rth_idx:], self.wsr_dpt[1][rth_idx:], 'k')
        ax[2, 0].plot(self.r[rth_idx:], self.wsr_dpt[2][rth_idx:], 'k')

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        s = np.array([1, 3, 5])
        scale_fac = np.sqrt((2*s + 1)/4./np.pi)
        unitconv = self.GVARS.OM * 1e9

        omega_dpt = []
        omega_fit = []
        fig1, axs1 = plt.subplots()
        for th in tqdm(theta, desc='Latitude plots'):
            legpoly = lpmn(0, 5, np.cos(th*np.pi/180.))[1][0, 1::2]
            omega_dpt.append((legpoly*scale_fac) @ self.wsr_dpt / self.r * unitconv)
            omega_fit.append((legpoly*scale_fac) @ wsr_spl_full / self.r * unitconv)

            fig, axs = plt.subplots()
            axs.plot(self.r[rth_idx:], omega_dpt[-1][rth_idx:],
                     'k', label='DPT', linewidth=0.7)
            axs.plot(self.r[rth_idx:], omega_fit[-1][rth_idx:],
                     '--r', label='Fit', linewidth=0.7)

            axs1.plot(self.r[rth_idx:], omega_dpt[-1][rth_idx:],
                     'k', label='DPT - $\\theta=$'+f'{th:.1f}', linewidth=0.7)
            axs1.plot(self.r[rth_idx:], omega_fit[-1][rth_idx:],
                     '--r', label='Fit - $\\theta=$'+f'{th:.1f}', linewidth=0.7)
            axs1.set_ylabel("$\\Omega(r)$")
            axs1.set_xlabel("$r$ in $R_{\odot}$", size=16)
            axs1.legend()

            axs.set_ylabel("$\\Omega(r)$")
            axs.set_xlabel("$r$ in $R_{\odot}$", size=16)
            axs.set_title(f"$\\Omega(r)$ at $\\theta=${th:.1f}")
            axs.legend()
            fig.savefig(f"{self.tag}_omega_th{th:04.1f}.pdf")
            plt.close(fig)
        fig1.savefig(f"{self.tag}_omega_all.pdf")
        plt.close(fig1)

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        # overplotting the reconstructed profile
        ax[0, 0].plot(self.r[rth_idx:], wsr_spl_full[0][rth_idx:], '--r', alpha=0.5)
        ax[1, 0].plot(self.r[rth_idx:], wsr_spl_full[1][rth_idx:], '--r', alpha=0.5)
        ax[2, 0].plot(self.r[rth_idx:], wsr_spl_full[2][rth_idx:], '--r', alpha=0.5)

        # settin axis labels and title
        ax[0, 0].set_title(f'$w_s(r)$ DPT vs. {self.tag}', size=16)
        ax[0, 0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1, 0].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_xlabel('$r$ in $R_{\odot}$', size=16)
        
        # plotting the error percentages
        w1r_errperc = self.get_percent_error(wsr_spl_full[0], self.wsr_dpt[0])
        w3r_errperc = self.get_percent_error(wsr_spl_full[1], self.wsr_dpt[1])
        w5r_errperc = self.get_percent_error(wsr_spl_full[2], self.wsr_dpt[2])
        
        ax[0, 1].semilogy(self.r[rth_idx:], abs(w1r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[1, 1].semilogy(self.r[rth_idx:], abs(w3r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[2, 1].semilogy(self.r[rth_idx:], abs(w5r_errperc)[rth_idx:], 'r', alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[2, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.tag}_wsr_zoom.pdf')
        plt.close()




    def get_percent_error(self, a1, a2):
        errperc = np.zeros_like(a1)
        diff = a1 - a2
        mask0 = abs(a1) < 1e-6
        errperc[~mask0] = abs(diff)[~mask0]*100/abs(a1)[~mask0]
        return errperc

    
