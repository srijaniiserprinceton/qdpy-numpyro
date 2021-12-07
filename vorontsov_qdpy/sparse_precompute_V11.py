import numpy as np
from tqdm import tqdm
from scipy import integrate
from scipy.interpolate import splev
from scipy import sparse
import sys

'''
from jax.experimental import sparse
from jax.ops import index_update as jidx_update
from jax.ops import index as jidx
import jax.numpy as jnp
from jax import jit
'''

from qdpy_jax import load_multiplets
from vorontsov_qdpy import prune_multiplets_V11
from qdpy_jax import jax_functions as jf
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import globalvars as gvar_jax
from vorontsov_qdpy import build_cenmult_and_nbs_V11 as build_cnm

# defining functions used in multiplet functions in the script
getnt4cenmult = build_cnm.getnt4cenmult
jax_minus1pow_vec = jf.jax_minus1pow_vec
_find_idx = wigmap.find_idx

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jf.jax_Omega
jax_gamma_ = jf.jax_gamma

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                    prune_multiplets_V11.get_pruned_attributes(GVARS,
                                                           GVARS_ST)

lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                     nl_idx_pruned,
                                     omega_pruned)

def get_bsp_basis_elements(x):
    """Returns the integrated basis polynomials
    forming the B-spline.

    Parameters
    ----------
    x : float, array-like
        The grid to be used for integration.

    bsp_params : A tuple containing (nc, t, k),
        where nc = the number of control points,
        t = the knot array from splrep,
        k = degree of the spline polynomials.
    """
    nc_total, t, k = GVARS.bsp_params
    basis_elements = np.zeros((GVARS.nc, len(x)))

    # looping over the basis elements for each control point
    for c_ind in range(GVARS.nc):
        # c = np.zeros(GVARS.ctrl_arr_dpt.shape[1])
        c = np.zeros_like(GVARS.ctrl_arr_dpt_full[0, :])
        c[GVARS.knot_ind_th + c_ind] = 1.0
        basis_elements[c_ind, :] = splev(x, (t, c, k))
    return basis_elements

# extracting the basis elements once 
bsp_basis = get_bsp_basis_elements(GVARS.r)

def build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s):
    # ls2fac
    ls2fac = ell1*(ell1+1) + ell2*(ell2+1) - s*(s+1)

    # slicing the required eigenfunctions
    U1, V1 = lm.U_arr[eig_idx1], lm.V_arr[eig_idx1]
    U2, V2 = lm.U_arr[eig_idx2], lm.V_arr[eig_idx2]
    
    # the factor in the integral dependent on eigenfunctions
    # shape (r,)
    eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac

    # total integrand
    # nc = number of control points, the additional value indicates the
    # integral between (rmin, rth), which is constant across MCMC iterations
    # shape (nc x r)
    integrand = -1. * bsp_basis * eigfac / GVARS.r

    # shape (nc,)
    post_integral = integrate.trapz(integrand, GVARS.r, axis=1)
    return post_integral

def integrate_fixed_wsr(eig_idx1, eig_idx2, ell1, ell2, s):
    s_ind = (s-1)//2
    ls2fac = ell1*(ell1+1) + ell2*(ell2+1) - s*(s+1)

    # slicing the required eigenfunctions
    U1, V1 = lm.U_arr[eig_idx1], lm.V_arr[eig_idx1]
    U2, V2 = lm.U_arr[eig_idx2], lm.V_arr[eig_idx2]

    # the factor in the integral dependent on eigenfunctions
    # shape (r,)
    eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
    # total integrand
    integrand = -1. * GVARS.wsr_fixed[s_ind] * eigfac / GVARS.r
    post_integral = integrate.trapz(integrand, GVARS.r) # a scalar
    return post_integral


def get_dim_hyper():
    """Returns the dimension of the hypermatrix
    dim_hyper = max(dim_super)
    """
    dim_hyper = 0
    nmults = len(GVARS.n0_arr)

    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)
        dim_super_local = np.sum(2*CENMULT_AND_NBS.nl_nbs[:, 1] + 1)
        if (dim_super_local > dim_hyper): dim_hyper = dim_super_local
    return dim_hyper


def build_SUBMAT_INDICES(CNM_AND_NBS):
    # supermatix can be tiled with submatrices corresponding to
    # (l, n) - (l', n') coupling. The dimensions of the submatrix
    # is (2l+1, 2l'+1)

    dim_blocks = len(CNM_AND_NBS.omega_nbs) # number of submatrix blocks along axis
    nl_nbs = np.asarray(CNM_AND_NBS.nl_nbs)
    dimX_submat = 2 * nl_nbs[:, 1] + 1

    # creating the startx, endx for submatrices
    submat_tile_ind = np.zeros((dim_blocks, 2), dtype='int32')

    for ix in range(0, dim_blocks):
        for iy in range(0, dim_blocks):
            submat_tile_ind[ix, 0] = np.sum(dimX_submat[:ix])
            submat_tile_ind[ix, 1] = np.sum(dimX_submat[:ix+1])

    # creating the submat-dictionary namedtuple
    SUBMAT_DICT = jf.create_namedtuple('SUBMAT_DICT',
                                       ['startx_arr',
                                        'endx_arr'],
                                       (submat_tile_ind[:, 0],
                                        submat_tile_ind[:, 1]))
    return SUBMAT_DICT


def build_hm_nonint_n_fxd_1cnm(CNM_AND_NBS, SUBMAT_DICT, dim_hyper, s):
    """Computes elements in the hypermatrix excluding the
    integral part.
    """
    # number of k's which are valid. See Vorontsov Eqn (33), (34)
    # smax=5 can couple a max of dell = 4 (needs to be even). 
    # (ell+k/2, ell-k/2) get coupled. So, coupling for k = 2, 4 only.
    num_k = (GVARS.smax - 1)//2

    # the non-c part of b_k_m
    noc_b_k_m = np.zeros((num_k, GVARS.nc, dim_hyper))

    # the fixed part of b_k_m (contribution below rth)
    fixed_b_k_m = np.zeros((num_k, dim_hyper))

    # extracting attributes from CNM_AND_NBS
    num_nbs = len(CNM_AND_NBS.omega_nbs)
    nl_nbs = CNM_AND_NBS.nl_nbs
    # omegaref = CNM_AND_NBS.omega_nbs[0]

    # extracting attributes from SUBMAT_DICT
    startx_arr, endx_arr = SUBMAT_DICT.startx_arr, SUBMAT_DICT.endx_arr
    # filling in the non-m part using the masks
    for i in range(num_nbs):
        # filling only Upper Triangle
        # for j in range(i, num_nbs):
        for j in range(num_nbs):
            ell1, ell2 = nl_nbs[i, 1], nl_nbs[j, 1]
            dell = ell1 - ell2
            dellx, delly = 0, 0

            if dell > 0:
                dellx = abs(dell)
                delly = 0
            elif dell < 0:
                dellx = 0
                delly = abs(dell)

            ellmin = min(ell1, ell2)

            # submat tiling indices
            startx, endx = startx_arr[i], endx_arr[i]
            starty, endy = startx_arr[j], endx_arr[j]

            wig1_idx, fac1 = _find_idx(ell1, s, ell2, 1)
            wigidx1ij = np.searchsorted(wig_idx, wig1_idx)
            wigval1 = fac1 * wig_list[wigidx1ij]

            m_arr = np.arange(-ellmin, ellmin+1)
            wig_idx_i, fac = _find_idx(ell1, s, ell2, m_arr)
            wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
            wigvalm = fac * wig_list[wigidx_for_s]
            
            # the mask matrix. Setting the elements in the relevant diagonal to 1                
            # this happens for all s where selection rule doesn't set it to zero
            # only using the odd-even selection rule here since the triangle rule has been 
            # taken care of in the creation of CNM and NBS
            mask_val = np.ones_like(m_arr) * (1 - jax_minus1pow_vec(ell1 + ell2 + s))
            mask_val = mask_val.astype('bool')
            
            np.fill_diagonal(mask_hypmat[startx+dellx:endx-dellx,
                                         starty+delly:endy-delly],
                             mask_val)

            #-------------------------------------------------------
            # computing the ell1, ell2 dependent factors such as
            # gamma and Omega
            gamma_prod = jax_gamma_(ell1) * jax_gamma_(ell2) * jax_gamma_(s) 
            Omega_prod = jax_Omega_(ell1, 0) * jax_Omega_(ell2, 0)
            
            # also including 4 pi (not including 2 * omegaref as per V11)
            ell1_ell2_fac = gamma_prod * Omega_prod *\
                            4 * np.pi * (1 - jax_minus1pow_vec(ell1 + ell2 + s))

            # parameters for calculating the integrated part
            eig_idx1 = nl_idx_pruned.index(CNM_AND_NBS.nl_nbs_idx[i])
            eig_idx2 = nl_idx_pruned.index(CNM_AND_NBS.nl_nbs_idx[j])

            # shape (n_control_points,)
            # integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
            integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
            #-------------------------------------------------------
            # integrating wsr_fixed for the fixed part
            fixed_integral = integrate_fixed_wsr(eig_idx1, eig_idx2, ell1, ell2, s)

            wigvalm *= (jax_minus1pow_vec(m_arr) * ell1_ell2_fac)

            wigprod = wigvalm * wigval1
            
            for c_ind in range(GVARS.nc):
                # non-ctrl points submat
                # avoiding  newaxis multiplication
                c_integral = integrated_part[c_ind] * wigprod
                np.fill_diagonal(non_c_hypmat_arr[c_ind, startx+dellx:endx-dellx,
                                                  starty+delly:endy-delly],
                                 c_integral)
    
            # the fixed hypermatrix
            f_integral = fixed_integral * wigprod
            np.fill_diagonal(fixed_hypmat[startx+dellx:endx-dellx,
                                          starty+delly:endy-delly],
                             f_integral)
    

    return non_c_hypmat_arr, fixed_hypmat, mask_hypmat


def build_hypmat_all_cenmults():
    # number of multiplets used
    nmults = len(GVARS.n0_arr)
    dim_hyper = get_dim_hyper()

    # storing as a list of sparse matrices
    # the fixed hypat (the part of hypermatrix that does not
    # change across iterations)
    fixed_hypmat_all_sparse = []
    noc_hypmat_all_sparse = []
    omegaref_nmults = []
    ell0_nmults = []

    # list to store all the sparse indices of all cenmults
    # to be used when reconverting back to dense before eigenvlue problem
    sp_indices_all = []


    # getting the sparse-element size for largest ell cenmult
    MAXMULT_AND_NBS = getnt4cenmult(GVARS.n0_arr[0], GVARS.ell0_arr[0], GVARS_ST)
    SUBMAT_DICT_MAX = build_SUBMAT_INDICES(MAXMULT_AND_NBS)
    __, __, maskmat_maxmult = build_hm_nonint_n_fxd_1cnm(MAXMULT_AND_NBS,
                                                         SUBMAT_DICT_MAX,
                                                         dim_hyper, GVARS.smax)
    
    # finding the sp_indices_maxmult
    maskmat_maxmult_sp = sparse.coo_matrix(maskmat_maxmult)
    sp_indices_maxmult = (maskmat_maxmult_sp.row, maskmat_maxmult_sp.col)
    
    # the shape of all the cenmult data and indices
    len_sp_indices_maxmult = len(sp_indices_maxmult[0])
    
    sp_indices_all.append(sp_indices_maxmult)

    # going over the cenmults in a reverse order
    # this is to ensure that the largest is fileld first
    # to fill the rest in the same max shape
    for i in tqdm(range(nmults), desc='nmult'):#-1, -1, -1):
        # looping over all the central multiplets                                      
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        ell0_nmults.append(ell0)

        # building the namedtuple for the central multiplet and its neighbours            
        CENMULT_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)
        SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
        omegaref_nmults.append(CENMULT_AND_NBS.omega_nbs[0])

        freqdiag = build_hypmat_freqdiag(CENMULT_AND_NBS,
                                         SUBMAT_DICT,
                                         dim_hyper)
        
        noc_hypmat_this_s = []
        
        for s_ind, s in enumerate(GVARS.s_arr):
            # shape (dim_hyper x dim_hyper) but sparse form
            non_c_hypmat, fixed_hypmat_s, maskmat_cenmult =\
                    build_hm_nonint_n_fxd_1cnm(CENMULT_AND_NBS,
                                               SUBMAT_DICT,
                                               dim_hyper, s)

            noc_hypmat_sparse_c_maxshaped = np.zeros((GVARS.nc,
                                                      len_sp_indices_maxmult))
            for c_ind in range(GVARS.nc):
                # the sparse data for the mask locations
                noc_hypmat_sparse_c = non_c_hypmat[c_ind, maskmat_cenmult]
                # enhancing the shape to maxmult
                noc_hypmat_sparse_c_maxshaped[c_ind, :len(noc_hypmat_sparse_c)] =\
                                                                noc_hypmat_sparse_c

            # appending the different m part in the list
            noc_hypmat_this_s.append(noc_hypmat_sparse_c_maxshaped)
            
            # adding up the different s for the fixed part
            if s_ind == 0:
                fixed_hypmat_this_mult = fixed_hypmat_s
            else:
                fixed_hypmat_this_mult += fixed_hypmat_s

        # getting the mask indices
        maskmat_cenmult_sp = sparse.coo_matrix(maskmat_cenmult)
        sp_indices_cenmult = (maskmat_cenmult_sp.row, maskmat_cenmult_sp.col)

        # adding the freqdiag to the fixed_hypmat
        fixed_hypmat_this_mult += np.diag(freqdiag)
        fixed_hypmat_this_mult_sparse = fixed_hypmat_this_mult[maskmat_cenmult]

        # making the shape compatible to the maxmult sparse form
        fixed_plus_freqdiag_maxshaped = np.zeros(len_sp_indices_maxmult)
        fixed_plus_freqdiag_maxshaped[:len(fixed_hypmat_this_mult_sparse)] =\
                                                        fixed_hypmat_this_mult_sparse
        # appending the sparse form of the fixed hypmat
        fixed_hypmat_all_sparse.append(fixed_plus_freqdiag_maxshaped)

        # appending the list of sparse matrices in s to the list in cenmults
        noc_hypmat_all_sparse.append(noc_hypmat_this_s)

        # storing the sparse indices for the particular central multiplet
        # if(i == nmults-1): continue
        if(i == 0): continue

        # adjusting the indices to the largest hypermatrix case (for no static slicing later)
        sp_indices_all.append(get_sp_indices_maxshaped(sp_indices_cenmult,
                                                       sp_indices_maxmult))
    
    # list of shape (nmults x s x (nc x dim_hyper, dim_hyper))
    # the last bracket denotes matrices of that shape but in sparse form
    return noc_hypmat_all_sparse, fixed_hypmat_all_sparse, \
        ell0_nmults, omegaref_nmults, sp_indices_all
