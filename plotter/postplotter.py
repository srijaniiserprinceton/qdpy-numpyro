import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmn
from tqdm import tqdm
import os

from qdpy import gen_wsr

plt.rcParams['axes.grid'] = True

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
plotdir_global = f"{scratch_dir}/plots"

class postplotter:
    def __init__(self, GVARS, ctrl_arr_fit_full, ctrl_arr_err_full, tag,
                 color='red', plotdir=plotdir_global, onlywsr=True):
        self.GVARS = GVARS
        self.r = GVARS.r
        self.OM = GVARS.OM
        self.wsr_dpt = GVARS.wsr
        self.wsr_err = GVARS.wsr_err
        self.ctrl_arr_dpt_full = ctrl_arr_fit_full
        self.ctrl_arr_err_full = ctrl_arr_err_full
        self.t_internal = GVARS.t_internal
        self.knot_ind_th = GVARS.knot_ind_th
        self.spl_deg = GVARS.spl_deg
        self.tag = tag
        self.plotdir = plotdir

        # plotting
        self.plot_fit_wsr()
        if not onlywsr:
            self.plot_fit_wsr_w_error()
            self.plot_fit_wsr_zoom()
            self.plot_omega_rtheta()

    def plot_fit_wsr(self, fig=None, ax=None, pcolor='red'):
        if fig == None and ax == None:
            fig, ax = plt.subplots(5, 2, figsize=(25, 7), sharex=True)

        lw = 0.5
        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r, self.wsr_dpt[0], 'k', linewidth=lw)
        ax[1, 0].plot(self.r, self.wsr_dpt[1], 'k', linewidth=lw)
        ax[2, 0].plot(self.r, self.wsr_dpt[2], 'k', linewidth=lw)
        ax[3, 0].plot(self.r, self.wsr_dpt[3], 'k', linewidth=lw)
        ax[4, 0].plot(self.r, self.wsr_dpt[4], 'k', linewidth=lw)

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.GVARS, self.r,
                                                   self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        # overplotting the reconstructed profile
        ax[0, 0].plot(self.r, wsr_spl_full[0], '--',
                      color=pcolor, alpha=0.5, linewidth=lw)
        ax[1, 0].plot(self.r, wsr_spl_full[1], '--',
                      color=pcolor, alpha=0.5, linewidth=lw)
        ax[2, 0].plot(self.r, wsr_spl_full[2], '--', 
                      color=pcolor, alpha=0.5, linewidth=lw)
        ax[3, 0].plot(self.r, wsr_spl_full[3], '--', 
                      color=pcolor, alpha=0.5, linewidth=lw)
        ax[4, 0].plot(self.r, wsr_spl_full[4], '--', 
                      color=pcolor, alpha=0.5, linewidth=lw)

        # settin axis labels and title
        ax[0, 0].set_title(f'$w_s(r)$ DPT vs. {self.tag}', size=16)
        ax[0, 0].set_ylabel('$w_1(r)$ in $\mu$Hz', size=16)
        ax[1, 0].set_ylabel('$w_3(r)$ in $\mu$Hz', size=16)
        ax[2, 0].set_ylabel('$w_5(r)$ in $\mu$Hz', size=16)
        ax[3, 0].set_ylabel('$w_7(r)$ in $\mu$Hz', size=16)
        ax[4, 0].set_ylabel('$w_9(r)$ in $\mu$Hz', size=16)
        ax[4, 0].set_xlabel('$r$ in $R_{\odot}$', size=16)
        
        # plotting the error percentages
        w1r_errperc = self.get_percent_error(wsr_spl_full[0], self.wsr_dpt[0])
        w3r_errperc = self.get_percent_error(wsr_spl_full[1], self.wsr_dpt[1])
        w5r_errperc = self.get_percent_error(wsr_spl_full[2], self.wsr_dpt[2])
        w7r_errperc = self.get_percent_error(wsr_spl_full[3], self.wsr_dpt[3])
        w9r_errperc = self.get_percent_error(wsr_spl_full[4], self.wsr_dpt[4])
        
        ax[0, 1].semilogy(self.r, abs(w1r_errperc), color=pcolor, alpha=0.5)
        ax[1, 1].semilogy(self.r, abs(w3r_errperc), color=pcolor, alpha=0.5)
        ax[2, 1].semilogy(self.r, abs(w5r_errperc), color=pcolor, alpha=0.5)
        ax[3, 1].semilogy(self.r, abs(w7r_errperc), color=pcolor, alpha=0.5)
        ax[4, 1].semilogy(self.r, abs(w9r_errperc), color=pcolor, alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[3, 1].set_ylabel('% offset in $w_7(r)$', size=14)
        ax[4, 1].set_ylabel('% offset in $w_9(r)$', size=14)
        ax[4, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[4, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.plotdir}/{self.tag}_wsr.pdf')
        plt.close()
        return fig, ax

    
    def plot_fit_wsr_w_error(self):
        fig, ax = plt.subplots(3, 2, figsize=(15, 7), sharex=True)
        
        err_spl_full = gen_wsr.get_wsr_from_spline(self.GVARS,
                                                   self.r, self.ctrl_arr_err_full,
                                                   self.t_internal, self.spl_deg)

        lw = 0.5
        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r, self.wsr_dpt[0], 'k', linewidth=lw)
        ax[1, 0].plot(self.r, self.wsr_dpt[1], 'k', linewidth=lw)
        ax[2, 0].plot(self.r, self.wsr_dpt[2], 'k', linewidth=lw)

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.GVARS,
                                                   self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        # converting to muHz
        # wsr_spl_full *= self.OM * 1e6
        # overplotting the reconstructed profile
        ax[0, 0].plot(self.r, wsr_spl_full[0], '--r', alpha=0.5, linewidth=lw)
        ax[1, 0].plot(self.r, wsr_spl_full[1], '--r', alpha=0.5, linewidth=lw)
        ax[2, 0].plot(self.r, wsr_spl_full[2], '--r', alpha=0.5, linewidth=lw)

        ax[0, 0].fill_between(self.r,
                              wsr_spl_full[0] - err_spl_full[0],
                              wsr_spl_full[0] + err_spl_full[0],
                              alpha=0.5, color='red')
        ax[1, 0].fill_between(self.r,
                              wsr_spl_full[1] - err_spl_full[1],
                              wsr_spl_full[1] + err_spl_full[1],
                              alpha=0.5, color='red')
        ax[2, 0].fill_between(self.r,
                              wsr_spl_full[2] - err_spl_full[2],
                              wsr_spl_full[2] + err_spl_full[2],
                              alpha=0.5, color='red')


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
        plt.savefig(f'{self.plotdir}/{self.tag}_wsr_with_error.pdf')
        plt.close()
    

    def plot_fit_wsr_zoom(self):
        fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)

        rth_idx = np.argmin(abs(self.r - self.GVARS.rth))

        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r[rth_idx:], self.wsr_dpt[0][rth_idx:], 'k')
        ax[1, 0].plot(self.r[rth_idx:], self.wsr_dpt[1][rth_idx:], 'k')
        ax[2, 0].plot(self.r[rth_idx:], self.wsr_dpt[2][rth_idx:], 'k')

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.GVARS,
                                                   self.r, self.ctrl_arr_dpt_full,
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
        
        ax[0, 1].semilogy(self.r[rth_idx:],
                          abs(w1r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[1, 1].semilogy(self.r[rth_idx:],
                          abs(w3r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[2, 1].semilogy(self.r[rth_idx:],
                          abs(w5r_errperc)[rth_idx:], 'r', alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[2, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.plotdir}/{self.tag}_wsr_zoom.pdf')
        plt.close()

    def plot_omega_rtheta(self, theta=np.arange(15, 105, 15)):
        fig, ax = plt.subplots(3, 3, figsize=(15, 7), sharex=True)
        err1d = self.GVARS.err1d[:, 1:-1]

        # rth_idx = np.argmin(abs(self.r - self.GVARS.rth - 0.1))
        rthcombined_idx = np.argmin(abs(self.r - 0.4))
        rth_idx = np.argmin(abs(self.r - 0.9))
        rlist = self.r[rth_idx:]
        # plot the wsr from dpt (no-spline)
        ax[0, 0].plot(self.r[rth_idx:], self.wsr_dpt[0][rth_idx:], 'k')
        ax[1, 0].plot(self.r[rth_idx:], self.wsr_dpt[1][rth_idx:], 'k')
        ax[2, 0].plot(self.r[rth_idx:], self.wsr_dpt[2][rth_idx:], 'k')

        # construct the spline from ctrl_arr_dpt_full
        wsr_spl_full = gen_wsr.get_wsr_from_spline(self.GVARS,
                                                   self.r, self.ctrl_arr_dpt_full,
                                                   self.t_internal, self.spl_deg)

        s = np.array([1, 3, 5])
        scale_fac = np.sqrt((2*s + 1)/4./np.pi)
        unitconv = self.GVARS.OM * 1e9

        omega_dpt = []
        omega_fit = []
        fig1, axs1 = plt.subplots()
        count = 0
        for th in tqdm(theta, desc='Latitude plots'):
            legpoly = lpmn(0, 5, np.cos(th*np.pi/180.))[1][0, 1::2]
            omega_dpt.append((legpoly*scale_fac) @ self.wsr_dpt / self.r * unitconv)
            omega_fit.append((legpoly*scale_fac) @ wsr_spl_full / self.r * unitconv)

            fig, axs = plt.subplots()
            axs.plot(self.r[rth_idx:], omega_dpt[-1][rth_idx:],
                     'r', label='DPT', linewidth=0.7)
            axs.fill_between(self.r[rth_idx:],
                             omega_dpt[-1][rth_idx:] - err1d[count][rth_idx:],
                             omega_dpt[-1][rth_idx:] + err1d[count][rth_idx:],
                             color='red', alpha=0.4)
            axs.plot(self.r[rth_idx:], omega_fit[-1][rth_idx:],
                     '--k', label='Fit', linewidth=0.7)

            axs1.plot(self.r[rthcombined_idx:], omega_dpt[-1][rthcombined_idx:],
                     'r', label='DPT - $\\theta=$'+f'{(90-th):.1f}', linewidth=0.7)
            axs1.fill_between(self.r[rthcombined_idx:],
                              omega_dpt[-1][rthcombined_idx:] -
                              err1d[count][rthcombined_idx:],
                              omega_dpt[-1][rthcombined_idx:] +
                              err1d[count][rthcombined_idx:],
                              color='red', alpha=0.4)
            axs1.plot(self.r[rthcombined_idx:], omega_fit[-1][rthcombined_idx:],
                     '--', label='Fit - $\\theta=$'+f'{(90-th):.1f}', linewidth=0.7)
            axs1.set_ylabel("$\\Omega(r)$")
            axs1.set_xlabel("$r$ in $R_{\odot}$", size=16)
            axs1.legend()

            axs.set_ylabel("$\\Omega(r)$")
            axs.set_xlabel("$r$ in $R_{\odot}$", size=16)
            axs.set_title(f"$\\Omega(r)$ at $\\theta=${(90-th):.1f}")
            axs.legend()
            fig.savefig(f"{self.plotdir}/{self.tag}_omega_th{(90-th):04.1f}.pdf")
            plt.close(fig)
        fig1.savefig(f"{self.plotdir}/{self.tag}_omega_all.pdf")
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
        
        ax[0, 1].semilogy(self.r[rth_idx:],
                          abs(w1r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[1, 1].semilogy(self.r[rth_idx:],
                          abs(w3r_errperc)[rth_idx:], 'r', alpha=0.5)
        ax[2, 1].semilogy(self.r[rth_idx:],
                          abs(w5r_errperc)[rth_idx:], 'r', alpha=0.5)
        
        # settin axis labels
        ax[0, 1].set_ylabel('% offset in $w_1(r)$', size=14)
        ax[1, 1].set_ylabel('% offset in $w_3(r)$', size=14)
        ax[2, 1].set_ylabel('% offset in $w_5(r)$', size=14)
        ax[2, 1].set_xlabel('$r$ in $R_{\odot}$', size=16)

        ax[2,1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.plotdir}/{self.tag}_wsr_zoom.pdf')
        plt.close()



    def get_percent_error(self, a1, a2):
        errperc = np.zeros_like(a1)
        diff = a1 - a2
        mask0 = abs(a1) < 1e-6
        errperc[~mask0] = abs(diff)[~mask0]*100/abs(a1)[~mask0]
        return errperc

    
