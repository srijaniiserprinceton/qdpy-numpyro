import numpy as np
import os
from collections import namedtuple
import jax.numpy as jnp

from qdpy_jax import load_multiplets

#----------------------------------------------------------------------
#                       All qts in CGS
# M_sol = 1.989e33 g
# R_sol = 6.956e10 cm
# B_0 = 10e5 G
# OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
# rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
#----------------------------------------------------------------------
filenamepath = os.path.realpath(__file__)
# taking [:-2] since we are ignoring the file name and current dirname
# this is specific to the way the directory structure is constructed
filepath = '/'.join(filenamepath.split('/')[:-2])   
configpath = filepath
with open(f"{configpath}/.config", "r") as f:
    dirnames = f.read().splitlines()

class qdParams():
    # {{{ Reading global variables
    # setting rmax as 1.2 because the entire r array needs to be used
    # in order to reproduce
    # (1) the correct normalization
    # (2) a1 = \omega_0 ( 1 - 1/ell ) scaling
    # (Since we are using lmax = 300, 0.45*300 \approx 150)
    
    # the radial orders present
    radial_orders = np.array([0], dtype='int32')
    # the bounds on angular degree for each radial order
    ell_bounds = np.array([[195, 210]], dtype='int32')
    
    rmin = 0.0
    rmax = 1.0
    rth = 0.98
    smax = 5
    fwindow =  150 
    precompute = False
    use_precomputed = False


class GlobalVars():
    """Class that initializes all the global variables
    just like in the original qdPy. However, the attributes
    are then split up into namedtuples depending on if we need
    it as a static or a traced namedtuple."""

    def __init__(self): 

        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais_dir = dirnames[2]
        self.datadir = f"{self.snrnmais_dir}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.eigdir = f"{self.snrnmais_dir}/eig_files"
        self.progdir = self.local_dir
        self.hmidata = np.loadtxt(f"{self.snrnmais_dir}/data_files/hmi.6328.36")


        datadir = f"{self.snrnmais_dir}/data_files"
        
        qdPars = qdParams()

        # Frequency unit conversion factor (in Hz (cgs))
        #all quantities in cgs
        M_sol = 1.989e33       # in grams
        R_sol = 6.956e10       # in cm
        B_0 = 10e5             # in Gauss (base of convection zone)
        self.OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol) 
        # should be 2.096367060263202423e-05 for above numbers

        # self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{datadir}/r.dat")
        self.nl_all = np.loadtxt(f"{datadir}/nl.dat").astype('int')
        self.nl_all_list = np.loadtxt(f"{datadir}/nl.dat").astype('int').tolist()
        self.omega_list = np.loadtxt(f"{datadir}/muhz.dat") * 1e-6 / self.OM

        # getting indices for minimum and maximum r
        if qdPars.precompute:
            self.rmin = 0.0
            self.rmax = rth
        elif qdPars.use_precomputed:
            self.rmin = qdPars.rth
            self.rmax = qdPars.rmax
        else:
            self.rmin = qdPars.rmin
            self.rmax = qdPars.rmax

        self.rmin_ind = self.get_ind(self.r, self.rmin)

        # removing the grid point corresponding to r=0
        # because Tsr has 1/r factor
        if self.rmin == 0:
            self.rmin_ind += 1
        self.rmax_ind = self.get_ind(self.r, self.rmax)
        

        self.smax = qdPars.smax
        self.fwindow = qdPars.fwindow

        # the rotation profile                                                                                                                                                        
        self.wsr = np.loadtxt(f'{self.datadir}/w_s/w.dat')

        # generating the multiplets which we will use
        n_arr, ell_arr = self.get_mult_arrays(load_from_file=False,
                                             radial_orders=qdPars.radial_orders,
                                             ell_bounds=qdPars.ell_bounds)

    
        # getting the pruned multiplets
        self.pruned_multiplets = load_multiplets.load_multiplets(self, n_arr, ell_arr)

        # not needed anymore
        del self.omega_list
        del self.nl_all
        del self.nl_all_list

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)
        self.wsr = self.mask_minmax(self.wsr)
        self.pruned_multiplets.U_arr = self.mask_minmax(self.pruned_multiplets.U_arr, axis=1)
        self.pruned_multiplets.V_arr = self.mask_minmax(self.pruned_multiplets.V_arr, axis=1)

        
        # the factor to be multiplied to make the upper and lower 
        # bounds of the model space to be explored
        self.fac_up = np.array([1.1, 2.0, 2.0])
        self.fac_lo = np.array([0.9, 0.0, 0.0])


        # converting to device array once
        '''
        self.wsr = jnp.array(self.wsr)   
        self.r = jnp.array(self.r)
        self.pruned_multiplets.nl_pruned = jnp.array(self.pruned_multiplets.nl_pruned)
        self.pruned_multiplets.omega_pruned = jnp.array(self.pruned_multiplets.omega_pruned)
        self.pruned_multiplets.U_arr = jnp.array(self.pruned_multiplets.U_arr)
        self.pruned_multiplets_V_arr = jnp.array(self.pruned_multiplets.V_arr)
        self.fac_up = jnp.array(self.fac_up)
        self.fac_lo = jnp.array(self.fac_lo)
        '''
        # rth = r threshold beyond which the profiles are updated. 
        self.rth = qdPars.rth
        
    def get_all_GVAR(self):
        '''Builds and returns the relevant dictionaries.
        At the location of this function call, the GVARS
        class instance containing all the other miscellaneous 
        arrays like nl_all and omega_list should be deleted.
        '''

        # the global path variables                                                                                                                                                              
        GVAR_PATHS = self.get_namedtuple_gvar_paths()
        # the global traced variables                                                                                                                                                            
        GVAR_TR = self.get_namedtuple_gvar_traced()
        # the global static variables                                                                                                                                                            
        GVAR_ST = self.get_namedtuple_gvar_static()
    
        # returns the relevant dictionaries
        return GVAR_PATHS, GVAR_TR, GVAR_ST 
        
    def get_mult_arrays(self, load_from_file=False, 
                        radial_orders=np.array([0]), ell_bounds=np.array([195, 210])):
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
            mults = np.load('qdpy_multiplets.npy')
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
        
        GVAR_PATHS_ = namedtuple('GVAR_PATHS', ['local_dir',
                                                'scratch_dir',
                                                'snrnmais_dir',
                                                'outdir',
                                                'eigdir',
                                                'progdir',
                                                'hmidata'])
        GVAR_PATHS = GVAR_PATHS_(self.local_dir,
                                 self.scratch_dir,
                                 self.snrnmais_dir,
                                 self.outdir,
                                 self.eigdir,
                                 self.progdir,
                                 self.hmidata)
        
        return GVAR_PATHS

    def get_namedtuple_gvar_traced(self):
        """Function to create the namedtuple containing the 
        various global attributes that can be traced by JAX.
        """
        
        GVAR_TRACED_ = namedtuple('GVAR_TRACED', ['r',
                                                  'rth',
                                                  'rmin_ind',
                                                  'rmax_ind',
                                                  'fac_up',
                                                  'fac_lo',
                                                  'nmults',
                                                  'wsr',
                                                  'U_arr',
                                                  'V_arr'])

        GVAR_TRACED = GVAR_TRACED_(self.r,
                                   self.rth,
                                   self.rmin_ind,
                                   self.rmax_ind,
                                   self.fac_up,
                                   self.fac_lo,
                                   len(self.pruned_multiplets.omega_pruned),
                                   self.wsr,
                                   self.pruned_multiplets.U_arr,
                                   self.pruned_multiplets.V_arr)

        return GVAR_TRACED
        
        
    def get_namedtuple_gvar_static(self):
        """Function to created the namedtuple containing the 
        various global attributes that are static arguments.
        """
        
        GVAR_STATIC_ = namedtuple('GVAR_STATIC', ['smax',
                                                  'nl_pruned',
                                                  'omega_pruned',
                                                  'fwindow',
                                                  'OM'])

        GVAR_STATIC = GVAR_STATIC_(self.smax,
                                   self.pruned_multiplets.nl_pruned,
                                   self.pruned_multiplets.omega_pruned,
                                   self.fwindow,
                                   self.OM)
        
        return GVAR_STATIC

    def get_ind(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr, axis=0):
        # if we want to clip the second axis (example in U_arr and V_arr)
        if(axis==1):
            return arr[:, self.rmin_ind:self.rmax_ind]
        else: return arr[self.rmin_ind:self.rmax_ind]
