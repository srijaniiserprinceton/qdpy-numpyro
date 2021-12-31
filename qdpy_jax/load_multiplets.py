import numpy as np
from tqdm import tqdm
from qdpy_jax import globalvars as gvar_jax


class load_multiplets:
    '''Picks out the multiplets and creates a  
    the list of multiplets and their original
    index in the naming convention of files.

    Parameters:
    -----------
    GVARS: dictionary
           Contains all the global variables created in globalvars.py.
    n_arr: array_like, int
           Array of radial orders for the multiplets.
    ell_arr: array_like, int
           Array of angular degree for the multiplets.
    '''

    def __init__(self, GVAR, nl_pruned, nl_idx_pruned, omega_pruned):
        self.GVAR = GVAR
        self.nl_pruned = nl_pruned
        self.omega_pruned = omega_pruned
        self.nl_idx_pruned = nl_idx_pruned
        self.U_arr = None
        self.V_arr = None
        self.load_eigs()
    
    def load_eigs(self):
        '''Loading the eigenfunctions only for the pruned multiplets.'''
        nmults = len(self.nl_pruned)
        rmin_idx = self.GVAR.rmin_ind
        rmax_idx = self.GVAR.rmax_ind

        U_arr = np.zeros((nmults, len(self.GVAR.r)))
        V_arr = np.zeros((nmults, len(self.GVAR.r)))

        # directory containing the eigenfunctions
        eigdir = self.GVAR.eigdir
        
        for i in tqdm(range(nmults), desc=f"Loading eigenfunctions..."):
            idx = self.nl_idx_pruned[i]
            U_arr[i] = np.loadtxt(f'{eigdir}/U{idx}.dat')[rmin_idx:rmax_idx]
            V_arr[i] = np.loadtxt(f'{eigdir}/V{idx}.dat')[rmin_idx:rmax_idx]

        self.U_arr = U_arr
        self.V_arr = V_arr
