import numpy as np   
from qdpy_jax import jax_functions as jf

def getnt4cenmult(n0, ell0, GVARS):
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

    # unperturbed frequency of central multiplet (n0, ell0)
    mult_idx = np.in1d(nl_arr[:, 0], n0) * np.in1d(nl_arr[:, 1], ell0)
    omega0 = omega_list[mult_idx]

    # frequency distances from central multiplet
    omega_diff = (omega_list - omega0) * GVARS.OM * 1e6
        
    # masking the other radial orders
    mask_n = abs(nl_arr[:,0] - n0) == 0 
    
    # masking the ells in the k band. V11 Eqn. (33)
    smax = GVARS.s_arr[-1]
    mask_k_width = abs(nl_arr[:,1] - ell0) <= (smax-1)//2

    # creating the final mask accounting for all of the masks above
    mask_nb_k = mask_n * mask_k_width

    # sorting the multiplets in ascending order of distance from (n0, ell0)
    sort_idx = np.argsort(abs(omega_diff[mask_nb_k]))
    
    # the final attributes that will be stored
    nl_neighbours = nl_arr[mask_nb_k][sort_idx]
    nl_neighbours_idx = nl_idx_vec(nl_neighbours)
    omega_neighbours = get_omega_neighbors(nl_neighbours_idx)

    CENMULT_AND_NBS = jf.create_namedtuple('CENMULT_AND_NBS',
                                           ['nl_nbs',
                                            'nl_nbs_idx',
                                            'omega_nbs'],
                                           (nl_neighbours,
                                            nl_neighbours_idx,
                                            omega_neighbours))

    return CENMULT_AND_NBS
