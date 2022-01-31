import numpy as np
from scipy import integrate
from scipy.interpolate import splev
import sys
import matplotlib.pyplot as plt

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

GVARS = gvar_jax.GlobalVars(n0=1,
                            lmin=72,
                            lmax=190)
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                    prune_multiplets.get_pruned_attributes(GVARS,
                                                           GVARS_ST)

lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                     nl_idx_pruned,
                                     omega_pruned)

# array containing different lower points in r
# starting from the surface and going deeper
r_lower_arr = np.linspace(0.8, 1, 100)[::-1]

r_lower_arr_ind = np.array([np.argmin(np.abs(GVARS.r - r_lp)) for r_lp in r_lower_arr])

# the location where kernel power will be compared
r_pow_comp = r_lower_arr[::20]
r_pow_comp_ind = np.array([np.argmin(np.abs(GVARS.r - r_pc)) for r_pc in r_pow_comp])


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
    integrand = -1. * GVARS.wsr[s_ind] * eigfac / GVARS.r

    # kern_pow
    kern_pow = np.abs(integrand[r_pow_comp_ind])

    post_integral_rlp = np.zeros_like(r_lower_arr_ind, dtype='float')

    # integrating uptil different lower limits of r
    for i, r_lp_ind in enumerate(r_lower_arr_ind):
        post_integral_rlp[i] +=\
        integrate.trapz(integrand[r_lp_ind:], GVARS.r[r_lp_ind:]) # a scalar
        
    return post_integral_rlp, kern_pow


def build_hm_nonint_n_fxd_1cnm(CNM_AND_NBS):
    """Computes elements in the hypermatrix excluding the
    integral part.
    """
    # extracting attributes from CNM_AND_NBS
    num_nbs = len(CNM_AND_NBS.omega_nbs)
    nl_nbs = CNM_AND_NBS.nl_nbs
    omegaref = CNM_AND_NBS.omega_nbs[0]

    # array to store the integral of coupled multiplets
    # as a function of the lower radius
    int_coupld_mult_rlp = []

    kern_pow = np.zeros_like(r_pow_comp)

    # filling in the non-m part using the masks
    for i in range(num_nbs):
        # filling only Upper Triangle
        # for j in range(i, num_nbs):
        omega_nl = CNM_AND_NBS.omega_nbs[i]
        
        for j in range(i, num_nbs):
            ell1, ell2 = nl_nbs[i, 1], nl_nbs[j, 1]
            
            # to store the sum over all s for that coupling
            int_coupld_mult_s = 0.0
            # looping over s and summing contrubution
            
            kern_pow_mult = np.zeros_like(r_pow_comp)
            for s_ind, s in enumerate(GVARS.s_arr):
    
                wig1_idx, fac1 = _find_idx(ell1, s, ell2, 1)
                wigidx1ij = np.searchsorted(wig_idx, wig1_idx)
                wigval1 = fac1 * wig_list[wigidx1ij]
                
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
                
                #-------------------------------------------------------
                # integrating wsr_fixed for the fixed part
                fixed_integral, kern_pow_cpl_mult_s =\
                            integrate_fixed_wsr(eig_idx1, eig_idx2, ell1, ell2, s)
                
                wigval1 *= ell1_ell2_fac
                
                int_coupld_mult_s += fixed_integral *  wigval1

                kern_pow_mult += kern_pow_cpl_mult_s

            # storing largest kernel powers
            get_val_ind = np.greater(kern_pow_mult, kern_pow)
            kern_pow[get_val_ind] = kern_pow_mult[get_val_ind]
    
            if(np.abs(np.sum(int_coupld_mult_s)) > 1e-6):
                # appending the integral (without m dependence)
                int_coupld_mult_rlp.append(int_coupld_mult_s)
                
    # shape is num of (unique couplings, r_lp)
    return np.asarray(int_coupld_mult_rlp), kern_pow


def build_hypmat_all_cenmults():
    # number of multiplets used
    nmults = len(GVARS.n0_arr)

    # which multiplet indices to loop over
    nmult_inds = np.arange(0, nmults, 10)

    fig, ax = plt.subplots(2, 1, figsize=(15,10), sharex=True)

    y_lower_lim = 0.0
    y_upper_lim = float(nmults)

    # starts with a shift of 0.5 and then keeps adding 1.0
    DC_shift = 0.0
    
    # array of lower turning point 
    r_ltp = []
    
    # kernel power array
    kern_pow = np.zeros((len(r_pow_comp), len(nmult_inds)))

    for i, mult_ind in enumerate(nmult_inds):
        print(i)
        # looping over all the central multiplets                                      
        n0, ell0 = GVARS.n0_arr[mult_ind], GVARS.ell0_arr[mult_ind]

        # building the namedtuple for the central multiplet and its neighbours            
        CENMULT_AND_NBS = getnt4cenmult(n0, ell0, GVARS_ST)                    

        # we are interested only in the absolute value
        integral_rlp, kern_pow_mult = build_hm_nonint_n_fxd_1cnm(CENMULT_AND_NBS)
        
        integral_rlp = np.abs(integral_rlp)

        ########## FOR THE FIRST PLOT #############
        # normalizing the integrals
        integral_rlp_norm = integral_rlp\
                / np.max(np.abs(integral_rlp), axis=1)[:, np.newaxis]
        
        # choosing just one of the couplings since they are similar
        integral_rlp_norm = integral_rlp_norm[0]

        # choosing the lower turning point by calculating gradient
        # gradient below 1e-6 is lower turning point
        r_ltp.append(r_lower_arr[::-1][np.argmax(np.where(integral_rlp_norm > (1.0 - 1e-3)))])

        # adjusting the lineplot in the row
        integral_rlp_norm += DC_shift

        ax[0].plot(r_lower_arr, integral_rlp_norm,
                 alpha=0.5)

        DC_shift += 1.0
        
        kern_pow[:,i] = kern_pow_mult
                             

    r_ltp = np.asarray(r_ltp)

    ax[0].plot(r_ltp, 1 + np.arange(len(r_ltp)), 'or')
    ax[0].set_xlim([r_lower_arr[-1], r_lower_arr[0]])
    ax[0].yaxis.set_ticklabels([])

    print(kern_pow.shape)
    # MAKING THE SECOND PLOT
    r_pow_comp_extd = np.append([1.0], r_pow_comp)
    for i in range(len(r_pow_comp)):
        kern_at_r = kern_pow[i]

        rel_pow_at_r = kern_at_r / np.max(kern_at_r)
        
        # compressing by 1/3rd of the gaps
        rel_pow_at_r *= np.diff(r_pow_comp_extd)[i]
        
        r_shift = r_pow_comp[i]

        ax[1].plot(rel_pow_at_r + r_shift, nmult_inds, '-ok')
        ax[1].axvline(r_shift, color='k')
        
    plt.tight_layout()
    plt.savefig('mult_rlp.pdf')

build_hypmat_all_cenmults()
