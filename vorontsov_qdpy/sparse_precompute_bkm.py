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
from vorontsov_qdpy import build_cenmult_and_nbs_bkm as build_cnm_bkm
from qdpy_jax import build_cenmult_and_nbs as build_cnm

# defining functions used in multiplet functions in the script
getnt4cenmult = build_cnm.getnt4cenmult
getnt4cenmult_bkm = build_cnm_bkm.getnt4cenmult

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
                                                               GVARS_ST,
                                                               # getnt4cenmult)
                                                               getnt4cenmult_bkm)

lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                     nl_idx_pruned,
                                     omega_pruned)

def get_max_lmax_and_nbs():
    nmults = len(GVARS.n0_arr)
    max_nbs = 0
    max_lmax = 0
    for i in range(nmults):
        n0 = GVARS.n0_arr[i]
        ell0 = GVARS.ell0_arr[i]
        CENMULT_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)
        num_nbs = len(CENMULT_AND_NBS.omega_nbs)
        max_nbs = max(num_nbs, max_nbs)
        max_lmax = max(max_lmax, max(CENMULT_AND_NBS.nl_nbs[:, 1]))

    return max_lmax, max_nbs

# necessary shapes for making arrays
max_lmax, max_nbs = get_max_lmax_and_nbs() 


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


def build_bkm_nonint_n_fxd_1cnm(CNM_AND_NBS, CNM_AND_NBS_bkm, k_arr, p_arr, s):
    """Computes elements in the hypermatrix excluding the
    integral part.
    """

    num_k = (GVARS.smax - 1)//2

    # the non-c part of b_k_m and
    # the fixed part of b_k_m (contribution below rth)
    noc_b_k_m  = np.zeros((GVARS.nc, num_k, 2*max_lmax+1))
    fixed_b_k_m = np.zeros((num_k, 2*max_lmax+1))

    # extracting attributes from CNM_AND_NBS
    num_nbs = len(CNM_AND_NBS_bkm.omega_nbs)
    nl_nbs_bkm = CNM_AND_NBS_bkm.nl_nbs
    nl_nbs = CNM_AND_NBS.nl_nbs


    # filling in the non-m part using the masks
    # k =  +/1 or k = +/2 are arranged one after another.
    # So, we can conveniently take steps of 2
    # for i in range(2, num_nbs, 2):
    for i in range(0, num_nbs, 2):
        j = i+1   # the index for the coupling multiplet

        # nl_nbs for CNM_AND_NBS_bkm is constructed such that k_ind = i//2                    
        k_ind = i//2
        
        ell1, ell2 = nl_nbs_bkm[i, 1], nl_nbs_bkm[j, 1]
        kval = ell2 - ell1
        ellmin = min(ell1, ell2)

        wig1_idx, fac1 = _find_idx(ell1, s, ell2, 1)
        wigidx1ij = np.searchsorted(wig_idx, wig1_idx)
        wigval1 = fac1 * wig_list[wigidx1ij]

        m_arr = np.arange(-ellmin, ellmin+1)
        lenm = len(m_arr)
        wig_idx_i, fac = _find_idx(ell1, s, ell2, m_arr)
        wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
        wigvalm = fac * wig_list[wigidx_for_s]

        #-------------------------------------------------------
        # computing the ell1, ell2 dependent factors such as
        # gamma and Omega
        gamma_prod = jax_gamma_(ell1) * jax_gamma_(ell2) * jax_gamma_(s) 
        Omega_prod = jax_Omega_(ell1, 0) * jax_Omega_(ell2, 0)
        
        # also including 4 pi (not including 2 * omegaref as per V11)
        ell1_ell2_fac = gamma_prod * Omega_prod *\
                        4 * np.pi * (1 - jax_minus1pow_vec(ell1 + ell2 + s))
        
        # parameters for calculating the integrated part
        eig_idx1 = nl_idx_pruned.index(CNM_AND_NBS_bkm.nl_nbs_idx[i])
        eig_idx2 = nl_idx_pruned.index(CNM_AND_NBS_bkm.nl_nbs_idx[j])
        
        # shape (n_control_points,)
        # integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
        integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
        #-------------------------------------------------------
        # integrating wsr_fixed for the fixed part
        fixed_integral = integrate_fixed_wsr(eig_idx1, eig_idx2, ell1, ell2, s)
        
        wigvalm *= (jax_minus1pow_vec(m_arr) * ell1_ell2_fac)
        wigprod = wigvalm * wigval1

        sidx = max_lmax - ellmin
        eidx = max_lmax + ellmin + 1
        
        for c_ind in range(GVARS.nc):
            # non-ctrl points submat
            # avoiding  newaxis multiplication
            c_integral = integrated_part[c_ind] * wigprod
            noc_b_k_m[c_ind, k_ind, sidx:eidx] = c_integral
            
        # the fixed hypermatrix
        # shape (k, 2*ellmin+1)
        f_integral = fixed_integral * wigprod
        fixed_b_k_m[k_ind, sidx:eidx] = f_integral
        k_arr[k_ind, sidx:eidx] = kval

    num_nbs = len(CNM_AND_NBS.omega_nbs)
    ell0 = nl_nbs[0, 1]
    for i in range(num_nbs):
        ell1 = nl_nbs[i, 1]
        p_arr[i] = ell1 - ell0

    return noc_b_k_m, fixed_b_k_m, k_arr, p_arr


def build_bkm_all_cenmults():
    # number of multiplets used
    nmults = len(GVARS.n0_arr)
    dim_hyper = get_dim_hyper()

    # number of k's which are valid. See Vorontsov Eqn (33), (34)
    # smax=5 can couple a max of dell = 4 (needs to be even). 
    # (ell+k/2, ell-k/2) get coupled. So, coupling for k = 2, 4 only.
    num_k = (GVARS.smax - 1)//2
    len_s = len(GVARS.s_arr)

    
    noc_bkm_shaped = np.zeros((nmults, len_s, GVARS.nc,
                               num_k, 2*max_lmax+1))
    fixed_bkm_shaped = np.zeros((nmults, num_k, 2*max_lmax+1))
    
    # to make it convenient to perform explicit k-dependent operations
    # to make it convenient to perform explicit p-dependent operations
    k_arr_shaped = np.zeros_like(fixed_bkm_shaped)
    p_arr_shaped = np.zeros((nmults, max_nbs))
    
    # looping over cenmtral multipelts
    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        
        # namedtuple for the central multiplet and its neighbours (V11) +
        # namedtuple for the central multiplet and its neighbours (supermatrix)
        CNM_AND_NBS_bkm = getnt4cenmult_bkm(n0, ell0, GVARS_ST)
        CNM_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)
        
        # list of arrays for different s
        noc_bkm_this_mult = []
        k_arr_local = np.zeros((num_k, 2*max_lmax+1))
        p_arr_local = np.zeros(max_nbs)

        # loop in s
        for sind, s in enumerate(GVARS.s_arr):
            # computing the unshaped bkm (noc and fixed)
            # build_bkm_nonint_n_fxd_1cnm(CNM_AND_NBS,
            noc_bkm, fixed_bkm, k_arr_local, p_arr_local = \
                build_bkm_nonint_n_fxd_1cnm(CNM_AND_NBS,
                                            CNM_AND_NBS_bkm,
                                            k_arr_local,
                                            p_arr_local,
                                            s)
            noc_bkm_shaped[i, sind, ...] = noc_bkm
            fixed_bkm_shaped[i, ...] += fixed_bkm

        k_arr_shaped[i, ...] = k_arr_local
        p_arr_shaped[i, ...] = p_arr_local

    return noc_bkm_shaped, fixed_bkm_shaped, k_arr_shaped, p_arr_shaped
