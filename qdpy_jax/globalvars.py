import numpy as np
import os

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
    rmin = 0.0
    rmax = 1.0
    rth = 0.98
    smax = 5
    fwindow =  150 
    # args = FN.create_argparser()
    n0 = 0
    l0 = 150
    precompute = False
    use_precomputed = False


class gvars_paths():
    def __init__(self):
        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais = dirnames[2]
        self.datadir = f"{self.snrnmais}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.eigdir = f"{self.snrnmais}/eig_files"
        self.progdir = self.local_dir
        self.hmidata = np.loadtxt(f"{self.snrnmais}/data_files/hmi.6328.36")

class globalVars():
    def __init__(self): #, rth=0.98, args=qdParams.args, rmin=qdPars.rmin,
                 #rmax=qdPars.rmax, smax=qdPars.smax, fwindow=qdPars.fwindow):

        datadir = f"{dirnames[2]}/data_files"
        
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

        self.rmin_idx = self.get_idx(self.r, self.rmin)

        # removing the grid point corresponding to r=0
        # because Tsr has 1/r factor
        if self.rmin == 0:
            self.rmin_idx += 1
        self.rmax_idx = self.get_idx(self.r, self.rmax)
        # print(f"rmin index = {self.rmin_idx}; rmax index = {self.rmax_idx}")

        self.smax = qdPars.smax
        self.fwindow = qdPars.fwindow

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)

        # rth = r threshold beyond which the profiles are updated. 
        self.rth = qdPars.rth
        
        self.fac_up = np.array([1.1, 2.0, 2.0])
        self.fac_lo = np.array([0.9, 0.0, 0.0])

        self.n0 = qdPars.n0
        self.l0 = qdPars.l0

    def get_idx(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr):
        return arr[self.rmin_idx:self.rmax_idx]
