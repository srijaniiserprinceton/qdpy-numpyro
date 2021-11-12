import numpy as np
import jax.numpy as jnp
from collections import namedtuple

from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS
from qdpy_jax import prune_multiplets
from qdpy_jax import wigner_map2 as wigmap

get_namedtuple_for_cenmult_and_neighbours =\
                    build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                        prune_multiplets.get_pruned_attributes(GVARS,
                                                                GVARS_ST)

def get_dim_hyper_and_num_nbs_total(GVARS, GVARS_ST):
    # the dimension of the hypermatrix
    dim_hyper = 0
    # total number of neighbours (all nbs for all multiplets)
    num_nbs_total = 0

    # total number of multiplets used
    nmults = len(GVARS.n0_arr)

    # running a dummy loop to know 
    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)

        # dim_super of local supermatrix                                                      
        dim_super = np.sum(2*CENMULT_AND_NBS.nl_nbs[:, 1] + 1)

        if(dim_super > dim_hyper): dim_hyper = dim_super

        num_nbs_total += len(CENMULT_AND_NBS.omega_nbs)

    return dim_hyper, num_nbs_total

def precompute(GVARS, GVARS_ST):
    # number of multiplets used
    nmults = len(GVARS.n0_arr)

    dim_hyper, num_nbs_total = get_dim_hyper_and_num_nbs_total(GVARS, GVARS_ST)
    nl_nbs_list = []
    nl_nbs_idx_list = []
    startx_list = []
    endx_list = []
    
    # array containing all the traces
    trace_arr = np.zeros((num_nbs_total, dim_hyper), dtype='bool')

    # array to store the number of neighbours for each central multiplet
    num_nbs_arr = np.zeros(nmults, dtype='int')

    # the cumulative neighbour count
    nbs_total_count = 0

    # looping over all the central multiplets
    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]

        # building the namedtuple for the central multiplet and its neighbours
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)

        # building the arrays for all nl_nbs and all omega_nbs
        if i == 0:
            nl_pruned_all = CENMULT_AND_NBS.nl_nbs
            omega_pruned_all = CENMULT_AND_NBS.omega_nbs
        else:
            nl_pruned_all = np.concatenate((nl_pruned_all, CENMULT_AND_NBS.nl_nbs), 0)
            omega_pruned_all = np.append(omega_pruned_all, CENMULT_AND_NBS.omega_nbs)

        num_nbs = len(CENMULT_AND_NBS.omega_nbs)
        nl_nbs_list.append(CENMULT_AND_NBS.nl_nbs)
        nl_nbs_idx_list.append(CENMULT_AND_NBS.nl_nbs_idx)
        num_nbs_arr[i] = num_nbs

        # building and storing the traces                                                  
        dimX_submat = 2 * CENMULT_AND_NBS.nl_nbs[:, 1] + 1

        startx_arr = np.cumsum(dimX_submat)[:-1]
        endx_arr = np.cumsum(dimX_submat)

        startx_arr = np.append([0], startx_arr)

        for tr_i in range(num_nbs):
            startx, endx = startx_arr[tr_i], endx_arr[tr_i]
            trace_arr[nbs_total_count, startx:endx] = 1
            nbs_total_count += 1

        startx_list.append(startx_arr)
        endx_list.append(endx_arr)

    # the start index of the neighbours of central multiplets
    nb_start_ind_arr = np.append([0], np.cumsum(num_nbs_arr)[:-1])
    nb_end_ind_arr = np.cumsum(num_nbs_arr)

    # the end index of the neighbours of the central multiplets
    # building the hypermatrix dictionary
    hm_dict_ = namedtuple('HM_DICT', ['dim_hyper',
                                      'trace_arr',
                                      'nb_start_ind_arr',
                                      'nb_end_ind_arr'])

    nl_dict_ = namedtuple('nl', ['nl_nbs',
                                 'startx_list',
                                 'endx_list'])
    nl_dict = nl_dict_(nl_nbs_list,
                       startx_list,
                       endx_list)

    HM_DICT = hm_dict_(dim_hyper,
                       jnp.asarray(trace_arr),
                       jnp.asarray(nb_start_ind_arr),
                       jnp.asarray(nb_end_ind_arr))

    return nl_pruned_all, omega_pruned_all, HM_DICT, nl_dict


def build_wig_hyper(cenmult_ind, hm_dict, nl_dict, s):
    _find_idx = wigmap.find_idx
    nl_nbs = nl_dict.nl_nbs[cenmult_ind]
    startx_arr = nl_dict.startx_list[cenmult_ind]
    endx_arr = nl_dict.endx_list[cenmult_ind]

    # the total dimension of the hypermatrix
    dim_hyper = hm_dict.dim_hyper

    # the non-m part of the hypermatrix
    m_hypmat = np.zeros((dim_hyper, dim_hyper))

    # fillin in the non-m part using the masks
    for i in range(len(nl_nbs)):
        for j in range(len(nl_nbs)):
            # filling in the diagonal blocks
            n1 = nl_nbs[i, 0]
            n2 = nl_nbs[j, 0]
            ell1 = nl_nbs[i, 0]
            ell2 = nl_nbs[j, 0]
            ellmin = min(ell1, ell2)
            m_arr = np.arange(-ellmin, ellmin+1)
            wig_idx_i, fac = _find_idx(ell1, s, ell2, m_arr)
            wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
            wigvals = fac * wig_list[wigidx_for_s]

            startx, endx = startx_arr[i], endx_arr[i]
            starty, endy = startx_arr[j], endx_arr[j]

            np.fill_diagonal(m_hypmat[startx:endx, starty:endy], wigvals)
    return m_hypmat
