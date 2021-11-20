from collections import namedtuple
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from numpy.polynomial.legendre import legval
import os
import matplotlib.pyplot as plt

# loading custom libraries/classes
from qdpy_jax import load_multiplets
from qdpy_jax import jax_functions as jf
from qdpy_jax import bsplines as bsp

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"
import sys
sys.path.append(f"{package_dir}/plotter")
import preplotter as preplotter

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

filenamepath = os.path.realpath(__file__)
filepath = '/'.join(filenamepath.split('/')[:-2])   
configpath = filepath
with open(f"{configpath}/.config", "r") as f:
    dirnames = f.read().splitlines()

class qdParams():
    # Reading global variables
    # setting rmax as 1.2 because the entire r array needs to be used
    # in order to reproduce
    # (1) the correct normalization
    # (2) a1 = \omega_0 ( 1 - 1/ell ) scaling
    # (Since we are using lmax = 300, 0.45*300 \approx 150)

    def __init__(self, lmin=200, lmax=200, n0=0):
        # the radial orders present
        self.radial_orders = np.array([n0])
        # the bounds on angular degree for each radial order
        # self.ell_bounds = np.array([[lmin, lmax]])
        self.ell_bounds = np.array([[lmin, lmax]])

        self.rmin, self.rth, self.rmax = 0.3, 0.9, 1.2
        self.fwindow =  150.0 
        self.smax = 5
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
                      "nl_all", "nl_all_list",
                      "omega_list", "fwindow",
                      "smax", "s_arr", "wsr",
                      "pruned_multiplets",
                      "INT",
                      "FLOAT"]

    __methods__ = ["get_all_GVAR",
                   "get_mult_arrays",
                   "get_namedtuple_gvar_paths",
                   "get_namedtuple_gvar_static",
                   "get_namedtuple_gvar_traced",
                   "get_ind", "mask_minmax"]

    def __init__(self, lmin=200, lmax=200, n0=0): 
        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais_dir = dirnames[2]
        self.datadir = f"{self.snrnmais_dir}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.eigdir = f"{self.snrnmais_dir}/eig_files"
        self.progdir = self.local_dir
        self.hmidata = np.loadtxt(f"{self.snrnmais_dir}/data_files/hmi.6328.36")

        datadir = f"{self.snrnmais_dir}/data_files"
        qdPars = qdParams(lmin=lmin, lmax=lmax, n0=n0)

        # Frequency unit conversion factor (in Hz (cgs))
        #all quantities in cgs
        M_sol = 1.989e33       # in grams
        R_sol = 6.956e10       # in cm
        B_0 = 10e5             # in Gauss (base of convection zone)
        self.OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol) 

        # self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{datadir}/r.dat").astype('float')
        self.nl_all = np.loadtxt(f"{datadir}/nl.dat").astype('int')
        self.nl_all_list = np.loadtxt(f"{datadir}/nl.dat").astype('int').tolist()
        self.omega_list = np.loadtxt(f"{datadir}/muhz.dat").astype('float')
        self.omega_list *= 1e-6 / self.OM

        self.rmin = qdPars.rmin
        self.rmax = qdPars.rmax

        # removing the grid point corresponding to r=0
        # because Tsr has 1/r factor
        self.rmin_ind = self.get_ind(self.r, self.rmin) + 1
        self.rmax_ind = self.get_ind(self.r, self.rmax)

        self.smax = qdPars.smax
        self.s_arr = np.arange(1, self.smax+1, 2)

        self.fwindow = qdPars.fwindow
        self.wsr = -1.0*np.loadtxt(f'{self.datadir}/w_s/w.dat')

        # self.wsr = np.ones_like(self.wsr) #!!
        # self.wsr = np.load(f'wsr-spline.npy')
        self.wsr_extend()

        
        # rth = r threshold beyond which the profiles are updated. 
        self.rth = qdPars.rth
        
        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        self.wsr = self.mask_minmax(self.wsr, axis=1)
        self.rth_ind = self.get_ind(self.r, self.rth)
        self.r_spline = self.r[self.rth_ind:]

        # generating the multiplets which we will use
        load_from_file = False
        n_arr, ell_arr = self.get_mult_arrays(load_from_file,
                                              qdPars.radial_orders,
                                              qdPars.ell_bounds)
        self.n0_arr, self.ell0_arr = n_arr, ell_arr
        self.eigvals_true, self.eigvals_sigma = self.get_eigvals_true()

        # the factor to be multiplied to make the upper and lower 
        # bounds of the model space to be explored
        self.fac_arr = np.array([[1.1, 1.9, 1.9],
                                 [0.9, 0.1, 0.1]])

        
        # finding the spline params for wsr
        self.spl_deg = 3
        self.knot_num = 100

        # getting  wsr_fixed and spline_coefficients
        bsplines = bsp.get_splines(self.r, self.rth, self.wsr,
                                   self.knot_num, self.fac_arr,
                                   self.spl_deg)
        self.wsr_fixed = bsplines.wsr_fixed
        self.ctrl_arr_up = bsplines.c_arr_up
        self.ctrl_arr_lo = bsplines.c_arr_lo
        self.ctrl_arr_dpt_clipped = bsplines.c_arr_dpt_clipped
        self.ctrl_arr_dpt_full = bsplines.c_arr_dpt_full
        self.t_internal = bsplines.t_internal
        self.knot_ind_th = bsplines.knot_ind_th
        
        self.bsp_params = (len(self.ctrl_arr_dpt_full),
                           self.t_internal,
                           self.spl_deg)
        self.nc = len(self.ctrl_arr_dpt_clipped[0])

        # throws an error if ctrl_arr_up is not always larger than ctrl_arr_lo
        np.testing.assert_array_equal([np.sum(self.ctrl_arr_lo>self.ctrl_arr_up)],[0])
        
        # converting necessary arrays to tuples
        self.s_arr = tuple(self.s_arr)
        self.omega_list= tuple(self.omega_list)
        self.nl_all = tuple(map(tuple, self.nl_all))

        # if preplot is True, plot the various things for
        # ensuring everything is working properly
        
        if qdPars.preplot:
            check_splines = preplotter.preplotter(self.r, self.OM, self.wsr,
                                                  self.wsr_fixed,
                                                  self.ctrl_arr_up,
                                                  self.ctrl_arr_lo,
                                                  self.ctrl_arr_dpt_full,
                                                  self.ctrl_arr_dpt_clipped,
                                                  self.t_internal,
                                                  self.knot_ind_th,
                                                  self.spl_deg)
        return None

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

    # {{{ def findfreq(data, l, n, m):
    def findfreq(self, l, n, m):
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

        data = self.hmidata
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
        split_sigmas = np.append([0.0], data[modeindex, 49:85])
        totsigma = compute_totsigma(split_sigmas, m, L)
        totsplit = legval(1.0*m/L, splits)*L*0.001
        totsplit[mask0] = 0
        amp[maskl] = 0
        return totsplit, totsigma, amp
    # }}} findfreq(data, l, n, m)


    def get_all_GVAR(self):
        '''Builds and returns the relevant dictionaries.
        At the location of this function call, the GVARS
        class instance containing all the other miscellaneous 
        arrays like nl_all and omega_list should be deleted.
        '''
        # getting the global traced (TR) and static (ST) variables
        GVAR_PATHS = self.get_namedtuple_gvar_paths()
        GVAR_TR = self.get_namedtuple_gvar_traced()
        GVAR_ST = self.get_namedtuple_gvar_static()
        return GVAR_PATHS, GVAR_TR, GVAR_ST 

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

        # loading from a file. Must be saved in the (nmults, 2) shape
        if(load_from_file):
            mults = np.load('qdpy_multiplets.npy').astype('int')
            n_arr, ell_arr = mults[:, 0], mults[:, 1]

        # creating the arrays when the ells are continuous in each radial orders
        else:
            for i, n in enumerate(radial_orders):
                ell_min, ell_max = ell_bounds[i]
                for ell in range(ell_min, ell_max+1):
                    n_arr = np.append(n_arr, n)
                    ell_arr = np.append(ell_arr, ell)

        return n_arr, ell_arr

    def get_namedtuple_gvar_paths(self):
        """Function to create the namedtuple containing the 
        various global paths for reading and writing files.
        """
        GVAR_PATHS = jf.create_namedtuple('GVAR_PATHS',
                                          ['local_dir',
                                           'scratch_dir',
                                           'snrnmais_dir',
                                           'outdir',
                                           'eigdir',
                                           'progdir',
                                           'hmidata'],
                                          (self.local_dir,
                                           self.scratch_dir,
                                           self.snrnmais_dir,
                                           self.outdir,
                                           self.eigdir,
                                           self.progdir,
                                           self.hmidata))
        return GVAR_PATHS

    def get_namedtuple_gvar_traced(self):
        """Function to create the namedtuple containing the 
        various global attributes that can be traced by JAX.
        """
        GVAR_TRACED = jf.create_namedtuple('GVAR_TRACED',
                                           ['r',
                                            'r_spline',
                                            'rth',
                                            'rmin_ind',
                                            'rmax_ind',
                                            'wsr',
                                            'ctrl_arr_up',
                                            'ctrl_arr_lo',
                                            'ctrl_arr_dpt_clipped',
                                            'knot_arr',
                                            'eigvals_true',
                                            'eigvals_sigma'],
                                           (self.r,
                                            self.r_spline,
                                            self.rth,
                                            self.rmin_ind,
                                            self.rmax_ind,
                                            self.wsr,
                                            self.ctrl_arr_up,
                                            self.ctrl_arr_lo,
                                            self.ctrl_arr_dpt_clipped,
                                            self.t_internal,
                                            self.eigvals_true,
                                            self.eigvals_sigma))
        return GVAR_TRACED

    def get_namedtuple_gvar_static(self):
        """Function to created the namedtuple containing the 
        various global attributes that are static arguments.
        """
        GVAR_STATIC = jf.create_namedtuple('GVAR_STATIC',
                                           ['s_arr',
                                            'nl_all',
                                            'omega_list',
                                            'fwindow',
                                            'OM',
                                            'rth_ind',
                                            'spl_deg'],
                                           (self.s_arr,
                                            self.nl_all,
                                            self.omega_list,
                                            self.fwindow,
                                            self.OM,
                                            self.rth_ind,
                                            self.spl_deg))

        return GVAR_STATIC

    def get_ind(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr, axis=0):
        # if we want to clip the second axis (example in U_arr and V_arr)
        if(axis==1):
            return arr[:, self.rmin_ind:self.rmax_ind]
        else:
            return arr[self.rmin_ind:self.rmax_ind]

    def get_wsr_spline_params(self, which_ex='upex'):
        # parameterizing in terms of cubic splines
        lenr = len(self.r_spline)
        r_spacing = int(lenr//self.knot_num)
        r_filtered = self.r_spline[::r_spacing]

        # removing end points because of requirement of splrep
        # endpoints are automatically padded up
        t_set = r_filtered[1:-1]
        self.spl_deg = 3

        t, c, k = splrep(self.r_spline, self.wsr[0, self.rth_ind:],
                         s=0, t=t_set, k=self.spl_deg)
        self.spl_deg = k

        # adjusting the zero-padding in c from splrep
        c = c[:-(k+1)]

        len_s = len(self.s_arr)
        c_arr = np.zeros((len_s, len(c)))

        for i in range(len_s):
            wsr_i_tapered = self.create_nearsurface_profile(i, which_ex=which_ex)
            t, c, __ = splrep(self.r_spline, wsr_i_tapered[self.rth_ind:],
                              s=0, t=t_set, k=self.spl_deg)
            # adjusting the zero-padding in c from splrep
            c = c[:-(k+1)]
            c_arr[i] = c

        return t, c_arr

    def get_spline_full_r(self, which_ex='upex'):
        # parameterizing in terms of cubic splines
        lenr = len(self.r)
        r_spacing = int(lenr//self.knot_num)
        r_filtered = self.r[::r_spacing]

        # removing end points because of requirement of splrep
        # endpoints are automatically padded up
        t_set = r_filtered[1:-1]
        self.spl_deg = 3

        t, c, k = splrep(self.r, self.wsr[0],
                         s=0, t=t_set, k=self.spl_deg)
        self.spl_deg = k

        # adjusting the zero-padding in c from splrep
        c = c[:-(k+1)]

        len_s = len(self.s_arr)
        c_arr = np.zeros((len_s, len(c)))

        for i in range(len_s):
            wsr_i_tapered = self.create_nearsurface_profile(i, which_ex=which_ex)
            t, c, __ = splrep(self.r, wsr_i_tapered,
                              s=0, t=t_set, k=self.spl_deg)
            # adjusting the zero-padding in c from splrep                                     
            c = c[:-(k+1)]
            c_arr[i] = c

        return t, c_arr

    def get_matching_function(self):
        return (np.tanh((self.r - self.rth - 0.07)/0.02) + 1)/2.0

    def create_nearsurface_profile(self, idx, which_ex='upex'):
        w_dpt = self.wsr[idx, :]
        w_new = np.zeros_like(w_dpt)
        matching_function = self.get_matching_function()

        if which_ex == 'upex':
            scale_factor = self.fac_up[idx]
        elif which_ex == 'loex':
            scale_factor = self.fac_lo[idx]
        else:
            return w_dpt

        # near surface enhanced or suppressed profile
        # & adding the complementary part below the rth
        w_new = matching_function * scale_factor\
                * w_dpt[np.argmax(np.abs(w_dpt))]\
                * np.ones_like(w_dpt)
        w_new += (1 - matching_function) * w_dpt
        return w_new

    def wsr_extend(self):
        r1ind = np.argmin(abs(self.r - 1))
        # x = self.r[r1ind-300:r1ind]
        for i in range(len(self.wsr)):
            # y = self.wsr[i, r1ind-300:r1ind]
            # f = interp1d(x, y, fill_value='extrapolate')
            # self.wsr[i, r1ind:] = f(self.r[r1ind:])
            self.wsr[i, r1ind:] = self.wsr[i, r1ind-1]

    def gen_wsr_from_c(self, x, bsp_params):
        t, c_arr, k = bsp_params
        wsr = np.zeros((len(self.s_arr), len(x)))
        
        for s_ind in range(len(self.s_arr)):
            c = c_arr[s_ind]
            wsr[s_ind] = splev(x, (t, c, k))

        return wsr
