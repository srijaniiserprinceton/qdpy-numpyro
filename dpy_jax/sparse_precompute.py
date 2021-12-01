import numpy as np
from scipy import integrate
from tqdm import tqdm
from scipy.interpolate import splev

from jax.experimental import sparse
import jax.numpy as jnp
from jax import jit

from dpy_jax import load_multiplets
from dpy_jax import prune_multiplets
from dpy_jax import jax_functions as jf
from dpy_jax import wigner_map2 as wigmap
from dpy_jax import globalvars as gvar_jax
from dpy_jax import build_cenmults as build_cnm

# defining functions used in multiplet functions in the script
getnt4cenmult = build_cnm.getnt4cenmult
_find_idx = wigmap.find_idx
jax_minus1pow_vec = jf.jax_minus1pow_vec

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jit(jf.jax_Omega)
jax_gamma_ = jit(jf.jax_gamma)

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                    prune_multiplets.get_pruned_attributes(GVARS,
                                                           GVARS_ST)

CNM = build_cnm.getnt4cenmult(GVARS)

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

def build_integrated_part(eig_idx, ell, s):
    # ls2fac
    ls2fac = 2*ell*(ell+1) - s*(s+1)

    # slicing the required eigenfunctions
    U, V = lm.U_arr[eig_idx], lm.V_arr[eig_idx]
    
    # the factor in the integral dependent on eigenfunctions
    # shape (r,)
    eigfac = 2*U*V - U**2 - 0.5*(V**2)*ls2fac

    # total integrand
    # nc = number of control points, the additional value indicates the
    # integral between (rmin, rth), which is constant across MCMC iterations
    # shape (nc x r)
    integrand = -1. * bsp_basis * eigfac / GVARS.r

    # shape (nc,)
    post_integral = integrate.trapz(integrand, GVARS.r, axis=1)
    return post_integral

def integrate_fixed_wsr(eig_idx, ell, s):
    s_ind = (s-1)//2
    ls2fac = 2*ell*(ell+1) - s*(s+1)

    # slicing the required eigenfunctions
    U, V = lm.U_arr[eig_idx], lm.V_arr[eig_idx]

    # the factor in the integral dependent on eigenfunctions
    # shape (r,)
    eigfac = 2*U*V - U**2 - 0.5*(V**2)*ls2fac
    # total integrand
    integrand = -1. * GVARS.wsr_fixed[s_ind] * eigfac / GVARS.r
    post_integral = integrate.trapz(integrand, GVARS.r) # a scalar
    return post_integral


def build_hm_nonint_n_fxd_1cnm(s):
    """Computes elements in the hypermatrix excluding the
    integral part.
    """
    two_ellp1_sum_all = np.sum(2 * CNM.nl_cnm[:, 1] + 1) 
    # the non-m part of the hypermatrix
    non_c_diag_arr = np.zeros((GVARS.nc, two_ellp1_sum_all))
    non_c_diag_list = []

    # the fixed hypermatrix (contribution below rth)
    fixed_diag_arr = np.zeros(two_ellp1_sum_all)

    # extracting attributes from CNM_AND_NBS
    num_cnm = len(CNM.omega_cnm)

    start_cnm_ind = 0

    # filling in the non-m part using the masks
    for i in tqdm(range(num_cnm), desc=f"Precomputing for s={s}"):
        # updating the start and end indices
        omega0 = CNM.omega_cnm[i]
        end_cnm_ind = np.sum(2 * CNM.nl_cnm[:i+1, 1] + 1)

        # self coupling for isolated multiplets
        ell = CNM.nl_cnm[i, 1]

        wig1_idx, fac1 = _find_idx(ell, s, ell, 1)
        wigidx1ij = np.searchsorted(wig_idx, wig1_idx)
        wigval1 = fac1 * wig_list[wigidx1ij]

        m_arr = np.arange(-ell, ell+1)
        wig_idx_i, fac = _find_idx(ell, s, ell, m_arr)
        wigidx_for_s = np.searchsorted(wig_idx, wig_idx_i)
        wigvalm = fac * wig_list[wigidx_for_s]

        #-------------------------------------------------------
        # computing the ell1, ell2 dependent factors such as
        # gamma and Omega
        gamma_prod =  jax_gamma_(s) * jax_gamma_(ell)**2  
        Omega_prod = jax_Omega_(ell, 0)**2
        
        # also including 8 pi * omegaref
        omegaref = CNM.omega_cnm[i]
        ell1_ell2_fac = gamma_prod * Omega_prod *\
                        8 * np.pi * omegaref *\
                        (1 - jax_minus1pow_vec(s))

        # parameters for calculating the integrated part
        eig_idx = nl_idx_pruned.index(CNM.nl_cnm_idx[i])

        # shape (n_control_points,)
        # integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
        integrated_part = build_integrated_part(eig_idx, ell, s)
        #-------------------------------------------------------
        # integrating wsr_fixed for the fixed part
        fixed_integral = integrate_fixed_wsr(eig_idx, ell, s)

        wigvalm *= (jax_minus1pow_vec(m_arr) * ell1_ell2_fac)

        for c_ind in range(GVARS.nc):
            # non-ctrl points submat
            non_c_diag_arr[c_ind, start_cnm_ind: end_cnm_ind] =\
                                    integrated_part[c_ind] * wigvalm * wigval1

        # the fixed hypermatrix
        fixed_diag_arr[start_cnm_ind: end_cnm_ind] = fixed_integral * wigvalm * wigval1 

        # updating the start index
        start_cnm_ind = end_cnm_ind 

    # deleting wigvalm 
    del wigvalm

    # making it a list to allow easy c * hypermat later
    for c_ind in range(GVARS.nc):
        non_c_diag_arr_sparse = sparse.BCOO.fromdense(non_c_diag_arr[c_ind])
        non_c_diag_list.append(non_c_diag_arr_sparse)

    del non_c_diag_arr # deleting for ensuring no extra memory

    # sparsifying the fixed hypmat
    fixed_diag_sparse = sparse.BCOO.fromdense(fixed_diag_arr)
    del fixed_diag_arr             

    return non_c_diag_list, fixed_diag_sparse


def build_hypmat_all_cenmults():

    # to store the cnm frequencies
    omega0_arr = np.zeros(np.sum(2 * CNM.nl_cnm[:,1] + 1))
    start_cnm_ind = 0
    for i, omega_cnm in enumerate(CNM.omega_cnm):
        # updating the start and end indices
        end_cnm_ind = np.sum(2 * CNM.nl_cnm[:i+1, 1] + 1)
        omega0_arr[start_cnm_ind:end_cnm_ind] = CNM.omega_cnm[i]

        # updating the start index
        start_cnm_ind = end_cnm_ind


    # stores the diags as a function of s and c. Shape (s x c) 
    non_c_diag_cs = []

    for s_ind, s in enumerate(GVARS.s_arr):
        # shape (dim_hyper x dim_hyper) but sparse form
        non_c_diag_s, fixed_diag_s =\
                    build_hm_nonint_n_fxd_1cnm(s)
        
        # appending the different m part in the list
        non_c_diag_cs.append(non_c_diag_s)
        
        # adding up the different s for the fixed part
        if s_ind == 0:
            fixed_diag = fixed_diag_s
        else:
            fixed_diag += fixed_diag_s
    # non_c_diag_s = (s x 2ellp1_sum_all), fixed_diag = (2ellp1_sum_all,)
    return non_c_diag_cs, fixed_diag, omega0_arr
