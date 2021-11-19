import numpy as np
from scipy import integrate
from scipy.interpolate import splev

from jax.experimental import sparse
from jax import jit

from qdpy_jax import load_multiplets
from qdpy_jax import prune_multiplets
from qdpy_jax import jax_functions as jf
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_cenmult_and_nbs as build_cnm

# defining functions used in multiplet functions in the script
getnt4cenmult = build_cnm.getnt4cenmult
_find_idx = wigmap.find_idx
jax_minus1pow_vec = jf.jax_minus1pow_vec

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jit(jf.jax_Omega)
jax_gamma_ = jit(jf.jax_gamma)

ARGS = np.loadtxt(".n0-lmin-lmax.dat", dtype=int)
GVARS = gvar_jax.GlobalVars(n0=ARGS[0],
                            lmin=ARGS[1],
                            lmax=ARGS[2])
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                    prune_multiplets.get_pruned_attributes(GVARS,
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


def build_hypmat_freqdiag(CNM_AND_NBS, SUBMAT_DICT, dim_hyper):
    freqdiag = np.zeros(dim_hyper)
    omegaref = CNM_AND_NBS.omega_nbs[0]
    for i in range(len(CNM_AND_NBS.omega_nbs)):
        omega_nl = CNM_AND_NBS.omega_nbs[i]
        startx = SUBMAT_DICT.startx_arr[i]
        endx = SUBMAT_DICT.endx_arr[i]
        freqdiag[startx:endx] += omega_nl**2 - omegaref**2
    return sparse.BCOO.fromdense(np.diag(freqdiag))


def build_hm_nonint_n_fxd_1cnm(CNM_AND_NBS, SUBMAT_DICT, dim_hyper, s):
    """Computes elements in the hypermatrix excluding the
    integral part.
    """
    # the non-m part of the hypermatrix
    non_c_hypmat_arr = np.zeros((GVARS.nc, dim_hyper, dim_hyper))
    non_c_hypmat_list = []

    # the fixed hypermatrix (contribution below rth)
    fixed_hypmat = np.zeros((dim_hyper, dim_hyper))

    # extracting attributes from CNM_AND_NBS
    num_nbs = len(CNM_AND_NBS.omega_nbs)
    nl_nbs = CNM_AND_NBS.nl_nbs
    omegaref = CNM_AND_NBS.omega_nbs[0]

    # extracting attributes from SUBMAT_DICT
    startx_arr, endx_arr = SUBMAT_DICT.startx_arr, SUBMAT_DICT.endx_arr
    # filling in the non-m part using the masks
    for i in range(num_nbs):
        # filling only Upper Triangle
        # for j in range(i, num_nbs):
        omega_nl = CNM_AND_NBS.omega_nbs[i]
        
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

            wig1_idx, fac1 = _find_idx(ell1, s, ell2, 1)
            wigidx1ij = np.searchsorted(wig_idx, wig1_idx)
            wigval1 = fac1 * wig_list[wigidx1ij]

            m_arr = np.arange(-ellmin, ellmin+1)
            wig_idx_i, fac = _find_idx(ell1, s, ell2, m_arr)
            wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
            wigvalm = fac * wig_list[wigidx_for_s]

            #-------------------------------------------------------
            # computing the ell1, ell2 dependent factors such as
            # gamma and Omega
            gamma_prod = jax_gamma_(ell1) * jax_gamma_(ell2) * jax_gamma_(s) 
            Omega_prod = jax_Omega_(ell1, 0) * jax_Omega_(ell2, 0)
            
            # also including 8 pi * omega_ref
            ell1_ell2_fac = gamma_prod * Omega_prod *\
                            8 * np.pi * omegaref *\
                            (1 - jax_minus1pow_vec(ell1 + ell2 + s))

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

            startx, endx = startx_arr[i], endx_arr[i]
            starty, endy = startx_arr[j], endx_arr[j]

            for c_ind in range(GVARS.nc):
                # non-ctrl points submat
                # avoiding  newaxis multiplication
                np.fill_diagonal(non_c_hypmat_arr[c_ind, startx+dellx:endx-dellx,
                                                  starty+delly:endy-delly],
                                 integrated_part[c_ind] * wigvalm * wigval1)
    
            # the fixed hypermatrix
            np.fill_diagonal(fixed_hypmat[startx+dellx:endx-dellx,
                                          starty+delly:endy-delly],
                            fixed_integral * wigvalm * wigval1)

    # deleting wigvalm 
    del wigvalm

    # making it a list to allow easy c * hypermat later
    for c_ind in range(GVARS.nc):
        non_c_hypmat_arr_sparse = sparse.BCOO.fromdense(non_c_hypmat_arr[c_ind])
        non_c_hypmat_list.append(non_c_hypmat_arr_sparse)

    del non_c_hypmat_arr # deleting for ensuring no extra memory

    # sparsifying the fixed hypmat
    fixed_hypmat_sparse = sparse.BCOO.fromdense(fixed_hypmat)
    del fixed_hypmat#, fixed_hypmat_UT

    return non_c_hypmat_list, fixed_hypmat_sparse


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

    for i in range(nmults):
        # looping over all the central multiplets                                      
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        ell0_nmults.append(ell0)

        # building the namedtuple for the central multiplet and its neighbours            
        CENMULT_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)
        SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
        omegaref_nmults.append(CENMULT_AND_NBS.omega_nbs[0])

        freqdiag_sparse = build_hypmat_freqdiag(CENMULT_AND_NBS,
                                                SUBMAT_DICT,
                                                dim_hyper)
        
        noc_hypmat_this_s = []
        
        for s_ind, s in enumerate(GVARS.s_arr):
            # shape (dim_hyper x dim_hyper) but sparse form
            non_c_hypmat, fixed_hypmat_s =\
                    build_hm_nonint_n_fxd_1cnm(CENMULT_AND_NBS,
                                               SUBMAT_DICT,
                                               dim_hyper, s)
            
            # appending the different m part in the list
            noc_hypmat_this_s.append(non_c_hypmat)
            
            # adding up the different s for the fixed part
            if s_ind == 0:
                fixed_hypmat_this_mult = fixed_hypmat_s
            else:
                fixed_hypmat_this_mult += fixed_hypmat_s

        # appending the sparse form of the fixed hypmat
        fixed_hypmat_all_sparse.append(fixed_hypmat_this_mult + freqdiag_sparse)

        # appending the list of sparse matrices in s to the list in cenmults
        noc_hypmat_all_sparse.append(noc_hypmat_this_s)

    # list of shape (nmults x s x (nc x dim_hyper, dim_hyper))
    # the last bracket denotes matrices of that shape but in sparse form
    return noc_hypmat_all_sparse, fixed_hypmat_all_sparse, ell0_nmults, omegaref_nmults
