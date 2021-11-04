import jax
import numpy as np   
# import jax.numpy as jnp
from collections import namedtuple
from functools import partial
import sys
import time

# imports from qdpy_jax
from qdpy_jax import jax_functions as jf

def get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS):
    """Function that returns the name tuple for the
    attributes of the central mode and the neighbours for 
    that central mode. n0 and ell0 are static since everything
    else depends on n0 and ell0."""

    def nl_idx(n0, ell0):
        try:
            idx = nl_list.index([n0, ell0])
        except ValueError:
            idx = None
            logger.error('Mode not found')
        return idx

    def nl_idx_vec(nl_neighbours):
        nlnum = nl_neighbours.shape[0]
        nlidx = np.zeros(nlnum, dtype='int32')
        for i in range(nlnum):
            nlidx[i] = nl_idx(nl_neighbours[i][0],
                              nl_neighbours[i][1])
        return nlidx

    def get_omega_neighbors(nl_idx):
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

    # defining various masks to minimize the multiplet-couplings
    # rejecting modes far in frequency
    mask_omega = abs(omega_diff) <= GVARS.fwindow 
    
    # rejecting modes that don't satisfy triangle inequality
    smax = GVARS.s_arr[-1]
    mask_ell = abs(nl_arr[:, 1] - ell0) <= smax

    # only even l1-l2 is coupled for odd-s rotation perturbation
    # this is specific to the fact that dr is considered for odd s only
    mask_odd = ((nl_arr[:, 1] - ell0)%2) == 0
    
    # creating the final mask accounting for all of the masks above
    mask_nb = mask_omega * mask_ell * mask_odd

    # sorting the multiplets in ascending order of distance from (n0, ell0)
    sort_idx = np.argsort(abs(omega_diff[mask_nb]))
    
    # the final attributes that will be stored
    nl_neighbours = nl_arr[mask_nb][sort_idx]
    nl_neighbours_idx = nl_idx_vec(nl_neighbours)
    omega_neighbours = get_omega_neighbors(nl_neighbours_idx)

    num_neighbours = len(nl_neighbours_idx)
    dim_super = (2*nl_neighbours[:, 1] + 1).sum()
    
    CENMULT_AND_NBS = jf.create_namedtuple('CENMULT_AND_NBS',
                                           ['nl_nbs',
                                            'nl_nbs_idx',
                                            'omega_nbs',
                                            'num_nbs',
                                            'dim_super'],
                                           (nl_neighbours,
                                            nl_neighbours_idx,
                                            omega_neighbours,
                                            num_neighbours,
                                            dim_super))

    return CENMULT_AND_NBS

    
