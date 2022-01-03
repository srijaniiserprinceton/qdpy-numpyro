import numpy as np   
from qdpy_jax import misc_functions
from qdpy_jax import cenmult_functions

def getnt4cenmult(CNM_AND_NBS, GVARS, return_shaped=False):
    """Function that returns the name tuple for the
    attributes of the central mode and the neighbours for 
    that central mode. n0 and ell0 are static since everything
    else depends on n0 and ell0.

    Parameters:
    -----------
    CNM_AND_NBS - namedtuple
        Namedtuple containing the varius attributes defining a 
        central multiplet in the qdpt supermatrix.
    GVARS - object
        Python object with the following attributes:
        nl_neighbours - np.ndarray(ndim=2, dtype=int)
        nl_all - np.ndarray(ndim=2, dtype=int)
        omega_list - np.ndarray(ndim=1, dtype=float)
        s_arr - np.ndarray(ndim=1, dtype=int)
        fwindow - float
        OM - float
    return_shaped - bool
        If the returned array should retain the shape (num_nbs, num_nbs, 2)

    Returns:
    --------
    CENMULT_AND_NBS - namedtuple containing 'nl_nbs', 'nl_nbs_idx', 'omega_nbs'
    """
    
    # initializing class containing functions needed for cenmult                             
    cnm_funcs = cenmult_functions.cenmult_functions(GVARS)

    omega_list = np.asarray(GVARS.omega_list)
    nl_arr = np.asarray(GVARS.nl_all)

    # here it makes the couplings needed in V11 Eqn.(26)
    n0, ell0 = CNM_AND_NBS.nl_nbs[0, :]  
    num_nbs = len(CNM_AND_NBS.nl_nbs[:, 1])

    # stores which two modes get couplied for each submatrix under
    # Taylor expansion in V11 Eqn.(26)
    M_couplings_nl = np.zeros((num_nbs, num_nbs, 2, 2), dtype='int')
    
    # looping over the submatrices
    for i, ellp in enumerate(CNM_AND_NBS.nl_nbs[:, 1]):
        for j, ellp_ in enumerate(CNM_AND_NBS.nl_nbs[:, 1]):
            p, p_ = ellp - ell0, ellp_ - ell0
            k = p - p_
            ell1, ell2 = ell0 + k//2 , ell0 - k//2
            M_couplings_nl[i, j, 0] = np.array([n0, ell1])
            M_couplings_nl[i, j, 1] = np.array([n0, ell2])

    # next we flatten the M_coupling_nl matrix
    # this is necessary to pass this through the same ops
    # in prune_multiplets and load_multiplets. We reshape later on.
    nl_neighbours_M = np.reshape(M_couplings_nl, (num_nbs * num_nbs * 2, 2))
    nl_neighbours_M_idx = cnm_funcs.nl_idx_vec(nl_neighbours_M)
    omega_neighbours_M = cnm_funcs.get_omega_neighbors(nl_neighbours_M_idx)

    if return_shaped:
        # reshaping to be used in the loop in sparse_precompute_M
        nl_neighbours_M = np.reshape(nl_neighbours_M, (num_nbs, num_nbs, 2, 2))
        nl_neighbours_M_idx = np.reshape(nl_neighbours_M_idx, (num_nbs, num_nbs, 2))
        omega_neighbours_M = np.reshape(omega_neighbours_M, (num_nbs, num_nbs, 2))

    CENMULT_AND_NBS = misc_functions.create_namedtuple('CENMULT_AND_NBS',
                                                       ['nl_nbs',
                                                        'nl_nbs_idx',
                                                        'omega_nbs'],
                                                       (nl_neighbours_M,
                                                        nl_neighbours_M_idx,
                                                        omega_neighbours_M))

    return CENMULT_AND_NBS
