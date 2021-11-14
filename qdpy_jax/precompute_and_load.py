import numpy as np
import jax.numpy as jnp
from jax import jit
from collections import namedtuple

from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS
from qdpy_jax import prune_multiplets
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import jax_functions as jf

# defining functions used in multiplet functions in the script
get_namedtuple_for_cenmult_and_neighbours =\
                    build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours
_find_idx = wigmap.find_idx
jax_minus1pow_vec = jf.jax_minus1pow_vec

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                        prune_multiplets.get_pruned_attributes(GVARS,
                                                                GVARS_ST)

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jit(jf.jax_Omega)
jax_gamma_ = jit(jf.jax_gamma)

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
    
    # m-part of the hypermatrix
    hypmat_m_part = np.zeros((3, nmults, dim_hyper, dim_hyper))

    # the cumulative neighbour count across cenmults
    nbs_cmlcount = [0]

    # counter for the nbs across multiplets
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
            # making the 1D masks
            startx, endx = startx_arr[tr_i], endx_arr[tr_i]
            trace_arr[nbs_total_count, startx:endx] = 1
            nbs_total_count += 1
            
        # storing the index needed for f_li_lj_s
        nbs_cmlcount.append(nbs_cmlcount[-1] + (num_nbs*(num_nbs+1)//2))

        startx_list.append(startx_arr)
        endx_list.append(endx_arr)

        # filling the m-part of the hypmat
        for s in GVARS.s_arr:
            s_idx = (s-1)//2
            hypmat_m_part[s_idx, i, :, :] = \
                build_hypmat_m_part(CENMULT_AND_NBS.nl_nbs,
                                    startx_arr,
                                    endx_arr,
                                    dim_hyper,
                                    s)

    # the start index of the neighbours of central multiplets
    nb_start_ind_arr = np.append([0], np.cumsum(num_nbs_arr)[:-1])
    nb_end_ind_arr = np.cumsum(num_nbs_arr)

    Gam_Om_Wig_Tsr = build_Gam_Om_Wig_Tsr(nmults, nl_nbs_list, GVARS.s_arr)

    # building the hypermatrix dictionary
    hm_dict_ = namedtuple('HM_DICT', ['dim_hyper',
                                      'trace_arr',
                                      'nb_start_ind_arr',
                                      'nb_end_ind_arr',
                                      'm_part',
                                      'GamOmWigTsr',
                                      'nbs_cmlcount'])

    HM_DICT = hm_dict_(dim_hyper,
                       jnp.asarray(trace_arr),
                       jnp.asarray(nb_start_ind_arr),
                       jnp.asarray(nb_end_ind_arr),
                       jnp.asarray(hypmat_m_part),
                       jnp.asarray(Gam_Om_Wig_Tsr),
                       jnp.asarray(nbs_cmlcount[:-1])) 
                       # unnecessary last point


    return nl_pruned_all, omega_pruned_all, HM_DICT


def build_hypmat_m_part(nl_nbs, startx_arr, endx_arr, dim_hyper, s):
    
    # the non-m part of the hypermatrix
    m_hypmat = np.zeros((dim_hyper, dim_hyper))

    # filling in the non-m part using the masks
    for i in range(len(nl_nbs)):
        for j in range(len(nl_nbs)):
            # filling in the diagonal blocks
            n1 = nl_nbs[i, 0]
            n2 = nl_nbs[j, 0]
            ell1 = nl_nbs[i, 1]
            ell2 = nl_nbs[j, 1]
            ellmin = min(ell1, ell2)
            m_arr = np.arange(-ellmin, ellmin+1)
            wig_idx_i, fac = _find_idx(ell1, s, ell2, m_arr)
            wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
            wigvals = fac * wig_list[wigidx_for_s]

            wigvals = wigvals * jax_minus1pow_vec(m_arr)

            startx, endx = startx_arr[i], endx_arr[i]
            starty, endy = startx_arr[j], endx_arr[j]

            np.fill_diagonal(m_hypmat[startx:endx, starty:endy], wigvals)
    return m_hypmat    

def build_Gam_Om_Wig_Tsr(nmults, nl_nbs_all, s_arr):
    '''Stores the following entries of submatrices for all cenmults
    x = stored, - = not stored
    
    x x x x x
    - x x x x
    - - x x x
    - - - x x 
    - - - - x

    for num_nbs of a cenmult being 5
    '''
    
    # total upper triangular (l1, l2) combination in a supermatrix
    # N(N+1)/2 where N = dim_blocks. 
    # Example: N = 5, then 5+4+3+2+1
    UT_num_nbs = np.array([len(nbs) * (len(nbs)+1) // 2 for nbs in nl_nbs_all]).sum()

    Gam_Om_Wig_Tsr = np.zeros((len(s_arr), UT_num_nbs))

    # counter for UT_num_nbs
    counter = 0
    for cenmult_ind in range(nmults):
        nl_nbs = nl_nbs_all[cenmult_ind]
        # iterating over the submatrices of different cenmults
        for i in range(len(nl_nbs)):
            # only upper triangle
            for j in range(i, len(nl_nbs)):
                # extracting the angular degrees
                ell1, ell2 = nl_nbs[i,1], nl_nbs[j,1]
                
                # forming the different factors dependent on l1 and l2
                Om1 = jax_Omega_(ell1, 0)
                Om2 = jax_Omega_(ell2, 0)
                gamma_ell1_ell2 = jax_gamma_(ell1) * jax_gamma_(ell2)
            
                for s_ind, s in enumerate(s_arr):
                    # computing the wigner
                    wig_idx_si, fac = _find_idx(ell1, s, ell2, 1)
                    wigidx_for_s = np.searchsorted(wig_idx, wig_idx_si)
                    wigval = fac * wig_list[wigidx_for_s]

                    gamma_s = jax_gamma_(s)
                    
                    prod_gammas = gamma_ell1_ell2 * gamma_s

                    Gam_Om_Wig_Tsr[s_ind, counter] =\
                        -(1 - jax_minus1pow_vec(ell1 + ell2 + s)) * \
                        Om1 * Om2 * wigval * prod_gammas * \
                        8*jnp.pi

                counter += 1
        
    return Gam_Om_Wig_Tsr
