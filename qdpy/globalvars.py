import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval
from scipy.interpolate import splrep, splev

#----------------------------------------------------------------------
# loading local libraries/classes
from qdpy import jax_functions as jf
from qdpy import bsplines as bsp
from plotter import preplotter as preplotter
#----------------------------------------------------------------------
import jax.numpy as jnp
NAX = np.newaxis
# SMAX_GLOBAL = 7

#----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
eigtype =  dirnames[-1].split('/')[-2].split('_')[1]
#----------------------------------------------------------------------
#                       All qts in CGS
#----------------------------------------------------------------------
# M_sol = 1.989e33 g
# R_sol = 6.956e10 cm
# B_0 = 10e5 G
# OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
# rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
# should be 2.096367060263202423e-05 for above numbers
#----------------------------------------------------------------------
# taking [:-2] since we are ignoring the file name and current dirname
# this is specific to the way the directory structure is constructed

class qdParams():
    # Reading global variables
    # setting rmax as 1.2 because the entire r array needs to be used
    # in order to reproduce
    # (1) the correct normalization
    # (2) a1 = \omega_0 ( 1 - 1/ell ) scaling
    # (Since we are using lmax = 300, 0.45*300 \approx 150)

    def __init__(self, lmin=200, lmax=200, n0=0, rth=0.9, smax_global=19):
        # the radial orders present
        self.radial_orders = np.array([n0])
        # the bounds on angular degree for each radial order
        # self.ell_bounds = np.array([[lmin, lmax]])
        self.ell_bounds = np.array([[lmin, lmax]])

        self.rmin, self.rth, self.rmax = 0.0, rth, 1.2
        self.fwindow =  150.0 
        self.smax_global = smax_global 
        self.preplot = True


class GlobalVars():
    """Class that initializes all the global variables
    just like in the original qdPy. However, the attributes
    are then split up into namedtuples depending on if we need
    it as a static or a traced namedtuple."""

    __attributes__ = ["local_dir", "scratch_dir",
                      "snrnmais_dir", "datadir",
                      "outdir", "eigdir", "progdir",
                      "hmidata",
                      "OM", "r",
                      "rmin", "rmax", "rmin_ind",
                      "nl_all", 
                      "omega_list", "fwindow",
                      "smax_global", "s_arr", "wsr",
                      "pruned_multiplets",
                      "INT",
                      "FLOAT"]

    __methods__ = ["get_mult_arrays",
                   "get_ind", "mask_minmax"]

    def __init__(self, lmin=200, lmax=200, n0=0, rth=0.9, knot_num=15,
                 load_from_file=0, relpath='.', instrument='hmi',
                 tslen=72, daynum=6328, numsplits=18, smax_global=19):

        # storing the parameters required for inversion
        self.tslen = tslen
        self.numsplits = numsplits
        self.instrument = instrument
        self.daynum = daynum

        # storing the directory structure
        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais_dir = dirnames[2]
        self.datadir = f"{self.snrnmais_dir}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.ipdir = f"{self.scratch_dir}/input_files"
        self.eigdir = f"{self.snrnmais_dir}/eig_files"
        self.progdir = self.local_dir

        fsuffix = f"{self.tslen}d.{self.daynum}.{self.numsplits}"
        self.filename_suffix = f"{self.instrument}.{fsuffix}"
        self.hmidata_in = np.loadtxt(f"{self.ipdir}/{self.instrument}/" +
                                     f"{self.instrument}.in.{fsuffix}")
        self.hmidata_out = np.loadtxt(f"{self.ipdir}/{self.instrument}/" +
                                      f"{self.instrument}.out.{fsuffix}")
        self.relpath = relpath
        self.eigtype = eigtype

        qdPars = qdParams(lmin=lmin, lmax=lmax, n0=n0, rth=rth, smax_global=smax_global)

        # Frequency unit conversion factor (in Hz (cgs))
        #all quantities in cgs
        M_sol = 1.989e33       # in grams
        R_sol = 6.956e10       # in cm
        B_0 = 10e5             # in Gauss (base of convection zone)
        self.OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
        self.rsun = R_sol

        self.r = np.loadtxt(f"{self.ipdir}/r.dat").astype('float')
        self.nl_all = np.loadtxt(f"{self.datadir}/nl.dat").astype('int')
        self.nl_all = tuple(map(tuple, self.nl_all))
        self.omega_list = np.loadtxt(f"{self.datadir}/muhz.dat").astype('float')
        self.omega_list *= 1e-6 / self.OM

        self.rmin = qdPars.rmin
        self.rmax = qdPars.rmax

        # removing the grid point corresponding to r=0
        # because Tsr has 1/r factor
        self.rmin_ind = self.get_ind(self.r, self.rmin) + 1
        self.rmax_ind = self.get_ind(self.r, self.rmax)

        self.smax_global = qdPars.smax_global
        self.s_arr = np.arange(1, self.smax_global+1, 2)
        self.sfactor = np.ones_like(self.s_arr)

        self.fwindow = qdPars.fwindow
        self.wsr = -1.0*np.loadtxt(f'{self.ipdir}/{self.instrument}/' +
                                   f'wsr.{self.instrument}.{fsuffix}')[:self.smax_global//2+1]
        # self.err1d = np.loadtxt(f'{self.ipdir}/err1d-{self.instrument}.dat')
        self.wsr_err = np.loadtxt(f'{self.ipdir}/{self.instrument}/' +
                                  f'err.{self.instrument}.{fsuffix}')[:self.smax_global//2+1]

        self.wsr = self.sfactor[:, NAX]*self.wsr
        self.wsr_err = self.sfactor[:, NAX]*self.wsr_err
        # self.wsr_extend()

        wsr_err = np.zeros_like(self.wsr)
        for i in range(self.wsr.shape[0]):
            wsr_err[i, :] = 0.10*abs(self.wsr[i]).max()*np.ones(self.wsr[0].shape)
        self.wsr_err = wsr_err

        # rth = r threshold beyond which the profiles are updated. 
        self.rth = qdPars.rth

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        self.wsr = self.mask_minmax(self.wsr, axis=1)
        self.wsr_err = np.abs(self.mask_minmax(self.wsr_err, axis=1))
        self.rth_ind = self.get_ind(self.r, self.rth)
        self.r_spline = self.r[self.rth_ind:]

        # generating the multiplets which we will use
        n_arr, ell_arr, omega0_arr = self.get_mult_arrays(load_from_file,
                                                          qdPars.radial_orders,
                                                          qdPars.ell_bounds)

        # sorting the ells form largest to smallest
        # irrespective of the n0 of the cenmult
        # This determines the dimension of the hypermatrix
        sorted_idx_ell = np.argsort(ell_arr)[::-1]
        # ell_arr, sorted_idx = np.unique(ell_arr, return_index=True)
        # sorted_idx_ell = sorted_idx[::-1]
        self.n0_arr = n_arr[sorted_idx_ell]
        self.ell0_arr = ell_arr[sorted_idx_ell]
        self.omega0_arr = omega0_arr[sorted_idx_ell]
        self.dom_dell = self.get_dom_dell()

        self.eigvals_true, self.eigvals_sigma = self.get_eigvals_true()
        self.acoeffs_true, self.acoeffs_sigma = self.get_acoeffs_true()
        self.acoeffs_out_HMI, self.acoeffs_sigma_out_HMI =\
                                            self.get_acoeffs_out_HMI()

        # the factor to be multiplied to make the upper and lower 
        self.fac_arr = np.ones((2, len(self.s_arr)))

        # finding the spline params for wsr
        self.spl_deg = 3
        self.knot_num = knot_num

        # getting  wsr_fixed and spline_coefficients
        bsplines = bsp.get_splines(self.r, self.rth, self.wsr,
                                   self.knot_num, self.fac_arr,
                                   self.spl_deg)
        self.wsr_fixed = bsplines.wsr_fixed
        self.ctrl_arr_up = bsplines.c_arr_up
        self.ctrl_arr_lo = bsplines.c_arr_lo
        self.ctrl_arr_dpt_clipped = bsplines.c_arr_dpt_clipped
        self.nc = len(self.ctrl_arr_dpt_clipped[0])
        self.ctrl_arr_dpt_full = bsplines.c_arr_dpt_full
        self.t_internal = bsplines.t_internal
        self.knot_locs = bsplines.knot_locs
        self.knot_ind_th = bsplines.knot_ind_th
        # all spline basis
        self.bsp_basis_full = bsplines.bsp_basis
        self.d_bsp_basis = bsplines.d_bsp_basis
        # np.save(f'{self.relpath}/bsp_basis_full.npy', self.bsp_basis_full)
        # only the splines basis corresponding to the ctrl_clipped
        self.bsp_basis = self.bsp_basis_full[-self.nc:]

        # getting  wsr_err spline_coefficients                                          
        bsplines_err = bsp.get_splines(self.r, self.rth, self.wsr_err,
                                       self.knot_num, self.fac_arr,
                                       self.spl_deg)
        self.ctrl_arr_sig_clipped = bsplines_err.c_arr_dpt_clipped

        # taking absolute value to avoid spurious negatives
        self.ctrl_arr_sig_full = np.abs(bsplines_err.c_arr_dpt_full)

        # throws an error if ctrl_arr_up is not always larger than ctrl_arr_lo
        np.testing.assert_array_equal([np.sum(self.ctrl_arr_lo>self.ctrl_arr_up)],[0])
        
        # converting necessary arrays to tuples
        self.s_arr = self.s_arr
        self.omega_list= tuple(self.omega_list)

        _GVARS = self
        # if preplot is True, plot the various things for
        # ensuring everything is working properly

        if qdPars.preplot:
            check_splines = preplotter.preplotter(_GVARS,
                                                  self.r, self.OM, self.wsr,
                                                  self.wsr_fixed,
                                                  self.ctrl_arr_up,
                                                  self.ctrl_arr_lo,
                                                  self.ctrl_arr_dpt_full,
                                                  self.ctrl_arr_dpt_clipped,
                                                  self.t_internal,
                                                  self.knot_ind_th,
                                                  self.wsr_err,
                                                  self.spl_deg)
        return None

    def get_dom_dell(self):
        n0arr = self.n0_arr
        ell0arr = self.ell0_arr
        dom_dell = []

        for i in range(len(ell0arr)):
            mult_ind = self.nl_all.index((n0arr[i], ell0arr[i]+1))
            omega1 = self.omega_list[mult_ind]
            
            if(omega1 == 0):
                mult_ind = self.nl_all.index((n0arr[i], ell0arr[i]-1))
                omega1 = self.omega_list[mult_ind]
                         
            omega0 = self.omega0_arr[i]
            dom_dell.append(abs(omega1 - omega0))
        return np.array(dom_dell)


    def get_eigvals_true(self):
        n0arr = self.n0_arr
        ell0arr = self.ell0_arr
        nmults = len(n0arr)
        eigvals_true = np.array([])
        eigvals_sigma = np.array([])
        for i in range(nmults):
            m = np.arange(-ell0arr[i], ell0arr[i]+1)
            _eval, _esig, __ = self.findfreq(ell0arr[i], n0arr[i], m)
            eigvals_true = np.append(eigvals_true, _eval)
            eigvals_sigma = np.append(eigvals_sigma, _esig)
        return eigvals_true, eigvals_sigma

    def get_acoeffs_true(self):
        n0arr = self.n0_arr
        ell0arr = self.ell0_arr
        nmults = len(n0arr)
        acoeffs_true = np.array([])
        acoeffs_sigma = np.array([])
        for i in range(nmults):
            _aval, _asig = self.find_acoeffs(self.hmidata_in, ell0arr[i], n0arr[i])
            acoeffs_true = np.append(acoeffs_true, _aval)
            acoeffs_sigma = np.append(acoeffs_sigma, _asig)
        return acoeffs_true*1e-3, acoeffs_sigma*1e-3

    def get_acoeffs_out_HMI(self):
        n0arr = self.n0_arr
        ell0arr = self.ell0_arr
        nmults = len(n0arr)
        acoeffs_true = np.array([])
        acoeffs_sigma = np.array([])
        for i in range(nmults):
            _aval, _asig = self.find_acoeffs(self.hmidata_out, ell0arr[i], n0arr[i])
            acoeffs_true = np.append(acoeffs_true, _aval)
            acoeffs_sigma = np.append(acoeffs_sigma, _asig)
        return acoeffs_true*1e-3, acoeffs_sigma*1e-3

    # {{{ def findfreq(self, l, n, m, fullfreq=False):
    def findfreq(self, l, n, m, fullfreq=False):
        '''
        Find the eigenfrequency for a given (l, n, m)
        using the splitting coefficients
        Inputs: (data, l, n, m)
            data - array (hmi.6328.36)
            l - harmonic degree
            n - radial order
            m - azimuthal order
        Outputs: (nu_{nlm}, fwhm_{nl}, amp_{nl})
            nu_{nlm}    - eigenfrequency in microHz
            fwhm_{nl} - FWHM of the mode in microHz
            amp_{nl}    - Mode amplitude (A_{nl})
        '''
        def compute_totsigma(sigmas, m, L):
            coeffs = np.zeros_like(sigmas)
            totsigma = np.zeros_like(m, dtype=np.float64)
            lencoeffs = len(coeffs)
            for i in range(lencoeffs):
                coeff1 = np.zeros_like(sigmas)
                coeff1[i] = 1
                leg = legval(1.0*m/L, coeff1)*L*0.001
                totsigma += (leg * sigmas[i])**2
            return np.sqrt(totsigma)

        data = self.hmidata_in
        L = np.sqrt(l*(l+1))
        try:
            modeindex = np.where((data[:, 0] == l) *
                                (data[:, 1] == n))[0][0]
        except IndexError:
            print(f"MODE NOT FOUND : l = {l:03d}, n = {n:03d}")
            modeindex = 0
        (nu, amp, fwhm) = data[modeindex, 2:5]
        amp = amp*np.ones(m.shape)
        mask0 = m == 0
        maskl = abs(m) >= l
        splits = np.append([0.0], data[modeindex, 12:48])
        split_sigmas = np.append([0.0], data[modeindex, 48:84])
        totsigma = compute_totsigma(split_sigmas, m, L)
        totsplit = legval(1.0*m/L, splits)*L*0.001
        totsplit[mask0] = 0
        amp[maskl] = 0
        if fullfreq:
            return nu+totsplit, totsigma, amp
        else:
            return totsplit, totsigma, amp
    # }}} findfreq(data, l, n, m)

    # {{{ def find_acoeffs(data, l, n):
    def find_acoeffs(self, data, l, n, odd=True):
        '''Find the splitting coefficients for a given (l, n) 
        in nHz

        Inputs: (data, l, n)
            data - array (hmi.6328.36)
            l - harmonic degree
            n - radial order

        Outputs: (a_{nl}, asigma_{nl})
            a_{nl}    - splitting coefficients in nHz
            asigma_{nl} - uncertainity
        '''
        L = np.sqrt(l*(l+1))
        try:
            modeindex = np.where((data[:, 0] == l) *
                                (data[:, 1] == n))[0][0]
        except IndexError:
            print(f"MODE NOT FOUND : l = {l:03d}, n = {n:03d}")
            modeindex = 0

        splits = np.append([0.0], data[modeindex, 12:48])
        split_sigmas = np.append([0.0], data[modeindex, 48:84])

        assert self.smax_global < len(splits), "smax > number of splitting coefficients"

        splits = splits[:self.smax_global+1]
        split_sigmas = split_sigmas[:self.smax_global+1]

        if odd:
            splits = splits[1::2]*self.sfactor
            split_sigmas = split_sigmas[1::2]*self.sfactor
            return splits, split_sigmas
        else:
            return splits, split_sigmas
    # }}} find_acoeffs(data, l, n)


    # {{{ def get_lower_tp(self, n, ell):
    def get_lower_tp(self, n, ell):
        """Returns the lower turning point of the given mode."""
        model_s_data = np.loadtxt(f"{self.ipdir}/modelS.dat")
        c = model_s_data[:, 1]
        r = model_s_data[:, 0]*self.rsun
        r[r==0] = 1.0
        omega = self.findfreq(n, ell,
                              np.array([0], dtype=int),
                              fullfreq=True)[0]*1e-6 #Hz
        rltp_idx = np.argmin(abs(c*c/r/r - omega*omega/ell/(ell+1)*4*np.pi*np.pi))
        return r[rltp_idx]/self.rsun
    # }}} get_lower_tp(self, n, ell)


    def get_mult_arrays(self, load_from_file, radial_orders, ell_bounds):
        '''Creates the n array and ell array. If discontonuous ell then load from a 
        pregenerated file.

        Parameters:
        -----------
        load_from_file: boolean
            Whether to just load n_arr and ell_arr from a pregenerated file.
        radial_orders: array_like
            An array of all the radial orders for which ell_arr will be generated.
        ell_bounds: array_like
            An array of bounds of angular degrees for each radial order in `radial_orders`.
        '''
        n_arr = np.array([], dtype='int32')
        ell_arr = np.array([], dtype='int32')
        omega0_arr = []

        # loading from a file. Must be saved in the (nmults, 2) shape
        if(load_from_file):
            mults = np.load(f'{self.relpath}/' +
                            f'qdpy_multiplets.{self.filename_suffix}.npy').astype('int')
            n_arr, ell_arr = mults[:, 0], mults[:, 1]
            for i in range(len(n_arr)):
                mult_ind = self.nl_all.index((n_arr[i], ell_arr[i]))
                omega0_arr.append(self.omega_list[mult_ind])

        # creating the arrays when the ells are continuous in each radial orders
        else:
            for i, n in enumerate(radial_orders):
                ell_min, ell_max = ell_bounds[i]
                for ell in range(ell_min, ell_max+1):
                    n_arr = np.append(n_arr, n)
                    ell_arr = np.append(ell_arr, ell)
                    mult_ind = self.nl_all.index((n, ell))
                    omega0_arr.append(self.omega_list[mult_ind])

        return n_arr, ell_arr, np.array(omega0_arr)


    def get_ind(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr, axis=0):
        # if we want to clip the second axis (example in U_arr and V_arr)
        if(axis==1):
            return arr[:, self.rmin_ind:self.rmax_ind]
        else:
            return arr[self.rmin_ind:self.rmax_ind]
