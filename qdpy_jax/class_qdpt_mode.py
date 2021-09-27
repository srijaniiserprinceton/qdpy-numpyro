import jax
import jax.numpy as np
from collections import namedtuple

class qdptMode:
    """Class that handles modes that are perturbed using QDPT. Each class instance                                                                                           
    corresponds to a central mode (l0, n0). The frequency space is scanned to find out                                                                      
    all the neighbouring modes (l, n) which interact with the central mode                                                                                             
    (and amongnst themselves). The supermatrix is constructed for all possible                                                                                   
    coupling combinations (l, n) <--> (l', n').                                                                                                                      
    """
    __all__ = ["nl_idx", "nl_idx_vec",
               "get_omega_neighbors",
               "get_mode_neighbors_params",
               "create_supermatrix",
               "update_supermatrix"]

    def __init__(self, gvar):
        # global variables are read from the main program                                                                                                               
        self.gvar = gvar

        # spline-dictionary is preloaded                                                                                                                                   
        self.n0 = gvar.n0
        self.l0 = gvar.l0
        self.smax = gvar.smax
        self.freq_window = gvar.fwindow

        # index (mode-catalog) corresponding to the central mode
        self.idx = self.nl_idx(self.n0, self.l0)
        self.omega0 = self.gvar.omega_list[self.idx]
        self.get_mode_neighbors_params()


# initializing the necessary variables in gvar
fwindow = 150   # in muHz
# central multiplet
n0, ell0 = 0, 200
# max degree of perturbation (considering odd only)
smax = 5
# hardcoding central multiplet index for (0, 200)
cenmult_idx = 3672 
# hardcong unperturbed freq of central multiplet
# needs to be scaled up by GVAR.OM * 1e6 to be in muHz
unit_omega = 20.963670602632025   # GVAR.OM * 1e6  
omega0 = 67.99455100807411 

# this dictionary does not change with changing central multiplets
GVAR = namedtuple('GVAR', 'fwindow smax unit_omega')

# this dictionary changes with central multiplet
# NEEDS TO BE A STATIC ARGUMENT
CENMULT = namedtuple('CENMULT', 'n0 ell0 omega0 cenmult_idx')

GVAR = GVAR(fwindow, smax, unit_omega)
CENMULT = CENMULT(n0, ell0, omega0, cenmult_idx)
