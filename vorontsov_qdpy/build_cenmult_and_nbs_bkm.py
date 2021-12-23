import numpy as np   
from qdpy_jax import misc_functions
from qdpy_jax import cenmult_functions

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
    # initializing class containing functions needed for cenmult
    cnm_funcs = cenmult_functions.cenmult_functions(GVARS)

    omega_list = np.asarray(GVARS.omega_list)
    nl_arr = np.asarray(GVARS.nl_all)

    # masking the other radial orders
    mask_n = abs(nl_arr[:, 0] - n0) == 0 
    
    # masking the ells in the k band. V11 Eqn. (33)
    smax = GVARS.s_arr[-1]
    mask_k_width = abs(nl_arr[:, 1] - ell0) <= (smax-1)//2

    # creating the final mask accounting for all of the masks above
    mask_nb_k = mask_n * mask_k_width

    # the unsorted array of nl_neighbours
    nl_neighbours_unsorted = nl_arr[mask_nb_k]
    # removing the central multiplet
    nl_neighbours_unsorted =\
        nl_neighbours_unsorted[np.abs(nl_neighbours_unsorted[:,1] - ell0) > 0]

    # sorting the multiplets in ascending order of distance from ell0 (i.e, k)
    # sort_idx = np.argsort(abs(omega_diff[mask_nb_k]))
    sort_idx = np.argsort(np.abs(nl_neighbours_unsorted[:,1] - ell0))
    
    # the final attributes that will be stored
    nl_neighbours = nl_neighbours_unsorted[sort_idx]
    nl_neighbours_idx = cnm_funcs.nl_idx_vec(nl_neighbours)
    omega_neighbours = cnm_funcs.get_omega_neighbors(nl_neighbours_idx)

    CENMULT_AND_NBS = misc_functions.create_namedtuple('CENMULT_AND_NBS',
                                                       ['nl_nbs',
                                                        'nl_nbs_idx',
                                                        'omega_nbs'],
                                                       (nl_neighbours,
                                                        nl_neighbours_idx,
                                                        omega_neighbours))

    return CENMULT_AND_NBS
