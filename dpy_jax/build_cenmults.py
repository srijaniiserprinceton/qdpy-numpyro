import numpy as np   
from qdpy import jax_functions as jf

def getnt4cenmult(GVARS):
    """Function that returns the name tuple for the
    attributes of the central mode and the neighbours for 
    that central mode. n0 and ell0 are static since everything
    else depends on n0 and ell0.

    Parameters:
    -----------
    n0 - int
        The radial order of the central multiplet
    ell0 - int
        The spherical harmonic degree of central multiplet
    GVARS - object
        Python object with the following attributes:
        nl_neighbours - np.ndarray(ndim=2, dtype=int)
        nl_all - np.ndarray(ndim=2, dtype=int)
        omega_list - np.ndarray(ndim=1, dtype=float)
        s_arr - np.ndarray(ndim=1, dtype=int)
        fwindow - float
        OM - float

    Returns:
    --------
    CENMULT_AND_NBS - namedtuple containing 'nl_nbs', 'nl_nbs_idx', 'omega_nbs'
    """
    def nl_idx(n0, ell0):
        """Find the index for given n0, ell0"""
        try:
            idx = nl_list.index([n0, ell0])
        except ValueError:
            idx = None
            logger.error('Mode not found')
        return idx

    def nl_idx_vec(nl_neighbours):
        """Find the index for given n0, ell0"""
        nlnum = nl_neighbours.shape[0]
        nlidx = np.zeros(nlnum, dtype='int32')
        for i in range(nlnum):
            nlidx[i] = nl_idx(nl_neighbours[i][0],
                              nl_neighbours[i][1])
        return nlidx

    def get_omega_neighbors(nl_idx):
        """Get omega of the neighbours of central multiplet"""
        nlnum = len(nl_idx)
        omega_neighbors = np.zeros(nlnum)
        for i in range(nlnum):
            omega_neighbors[i] = GVARS.omega_list[nl_idx[i]]
        return omega_neighbors

    omega_list = np.asarray(GVARS.omega_list)
    nl_arr = np.asarray(GVARS.nl_all)
    nl_list = list(map(list, GVARS.nl_all))
    
    # the final attributes that will be stored
    nl_cnm = np.zeros((len(GVARS.n0_arr), 2), dtype='int')
    nl_cnm[:, 0] = GVARS.n0_arr
    nl_cnm[:, 1] = GVARS.ell0_arr
    nl_cnm_idx = nl_idx_vec(nl_cnm)
    omega_cnm = omega_list[nl_cnm_idx]

    CENMULTS = jf.create_namedtuple('CENMULT',
                                    ['nl_cnm',
                                     'nl_cnm_idx',
                                     'omega_cnm'],
                                    (nl_cnm,
                                     nl_cnm_idx,
                                     omega_cnm))

    return CENMULTS
