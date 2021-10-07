import numpy as np   
from collections import namedtuple
from functools import partial
import sys
import time
# imports from qdpy_jax
from qdpy_jax import globalvars

#------((( creating the namedtuples of global variables --------

GVARS = globalvars.GlobalVars()
# the global path variables
GVARS_PATHS = GVARS.get_namedtuple_gvar_paths()
# the global traced variables
GVARS_TR = GVARS.get_namedtuple_gvar_traced()
# the global static variables 
GVARS_ST = GVARS.get_namedtuple_gvar_static()

#------- creating the namedtuples of global variables )))-------

#--------------((( creating qdptMode namedtuple ----------------

def nl_idx(n0, l0):
    try:
        idx = GVARS_TR.nl_all_list.index([n0, l0])
    except ValueError:
        idx = None
        logger.error('Mode not found')
    return idx

def nl_idx_vec(nl_list):
    nlnum = nl_list.shape[0]
    nlidx = np.zeros(nlnum, dtype='int32')
    for i in range(nlnum):
        nlidx[i] = nl_idx(nl_list[i][0],
                          nl_list[i][1])
    return nlidx

def get_omega_neighbors(nl_idx):
    nlnum = len(nl_idx)
    omega_neighbors = np.zeros(nlnum)
    for i in range(nlnum):
        omega_neighbors[i] = GVARS_TR.omega_list[nl_idx[i]]
    return omega_neighbors

def get_namedtuple_for_cenmult_and_neighbours(n0, ell0):
    """Function that returns the name tuple for the
    attributes of the central mode and the neighbours for 
    that central mode. n0 and ell0 are static since everything
    else depends on n0 and ell0."""

    # unperturbed frequency of central multiplet (n0, ell0)
    mult_idx = GVARS_TR.nl_all_list.index([n0, ell0])
    omega0 = GVARS_TR.omega_list[mult_idx]

    
    omega_list = GVARS_TR.omega_list
    nl_all = GVARS_TR.nl_all
    omega_diff = (omega_list - omega0) * GVARS_TR.OM * 1e6

    # defining various masks to minimize the multiplet-couplings
 
    # rejecting modes far in frequency
    mask_omega = abs(omega_diff) <= GVARS_TR.fwindow 
    
    # rejecting modes that don't satisfy triangle inequality
    mask_ell = abs(nl_all[:, 1] - ell0) <= GVARS_ST.smax

    # only even l1-l2 is coupled for odd-s rotation perturbation
    # this is specific to the fact that DR is considered for odd s only
    mask_odd = ((GVARS_TR.nl_all[:, 1] - ell0)%2) == 0
    
    # creating the final mask accounting for all of the masks above
    mask_nb = mask_omega * mask_ell * mask_odd

    # sorting the multiplets in ascending order of distance from (n0, ell0)
    sort_idx = np.argsort(abs(omega_diff[mask_nb]))
    
    # the final attributes that will be stored
    nl_neighbours = nl_all[mask_nb][sort_idx]
    nl_neighbours_idx = nl_idx_vec(nl_neighbours)
    omega_neighbours = get_omega_neighbors(nl_neighbours_idx)
    num_neighbours = len(nl_neighbours_idx)
    
    dim_super = 2 * np.sum(nl_neighbours[:,1] + 1)
    dim_blocks = np.size(omega_neighbours)

    # creating the namedtuple
    CENMULT_AND_NBS_ = namedtuple('CENMULT_AND_NBS', ['nl_nbs',
                                                      'nl_nbs_idx',
                                                      'omega_nbs',
                                                      'num_nbs',
                                                      'dim_blocks',
                                                      'dim_super'])
    
    CENMULT_AND_NBS = CENMULT_AND_NBS_(nl_neighbours,
                                       nl_neighbours_idx,
                                       omega_neighbours,
                                       num_neighbours,
                                       dim_blocks,
                                       dim_super)
    
    return CENMULT_AND_NBS

