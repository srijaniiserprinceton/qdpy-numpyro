import numpy as np

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

    def __init__(self, GVAR, n_arr, ell_arr):
        
        self.GVAR = GVAR
        self.nl_pruned = None
        self.omega_pruned = None
        self.U_arr = None
        self.V_arr = None
        
        self.prune_multiplets(n_arr, ell_arr)
        self.load_eigs()


    def prune_multiplets(self, n_arr, ell_arr):
        nl_all = self.GVAR.nl_all
        # building a mask to prune only the desired multiplets
        mask_mults = np.in1d(nl_all[:, 0], n_arr) * np.in1d(nl_all[:, 1], ell_arr)

        # the desired multiplets 
        self.nl_pruned = nl_all[mask_mults]
        self.omega_pruned = self.GVAR.omega_list[mask_mults]
        
    
    def load_eigs(self):
        '''Loading the eigenfunctions only for the pruned multiplets.
        '''
        nmults = len(self.nl_pruned)

        print(self.nl_pruned.shape)

        U_arr = np.zeros((nmults, len(self.GVAR.r)))
        V_arr = np.zeros((nmults, len(self.GVAR.r)))

        # directory containing the eigenfunctions
        eigdir = self.GVAR.eigdir

        for i in range(nmults):
            n, ell = self.nl_pruned[i,0], self.nl_pruned[i,1]
            idx = self.GVAR.nl_all_list.index([n, ell])

            U_arr[i] = np.loadtxt(f'{eigdir}/U{idx}.dat')
            V_arr[i] = np.loadtxt(f'{eigdir}/U{idx}.dat')

        # clipping the eigenfunction to the desired radius range
        self.U_arr = U_arr[self.GVAR.rmin_ind: self.GVAR.rmax_ind]
        self.V_arr = V_arr[self.GVAR.rmin_ind: self.GVAR.rmax_ind]
