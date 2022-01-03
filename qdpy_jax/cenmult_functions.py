import numpy as np

class cenmult_functions:
    def __init__(self, GVARS):
        self.GVARS = GVARS

    def nl_idx(self, n0, ell0):
        """Find the index for given n0, ell0"""
        try:
            idx = self.GVARS.nl_all.index((n0, ell0))
        except ValueError:
            idx = None
            logger.error('Mode not found')
        return idx
        
    def nl_idx_vec(self, nl_neighbours):
        """Find the index for given n0, ell0"""
        nlnum = nl_neighbours.shape[0]
        nlidx = np.zeros(nlnum, dtype='int32')
        for i in range(nlnum):
            nlidx[i] = self.nl_idx(nl_neighbours[i][0],
                              nl_neighbours[i][1])
        return nlidx
        
    def get_omega_neighbors(self, nl_idx):
        """Get omega of the neighbours of central multiplet"""
        nlnum = len(nl_idx)
        omega_neighbors = np.zeros(nlnum)
        for i in range(nlnum):
            omega_neighbors[i] = self.GVARS.omega_list[nl_idx[i]]
        return omega_neighbors
