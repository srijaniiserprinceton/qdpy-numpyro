import jax.numpy as jnp
from jax.ops import index_add as jidx_add
from jax.ops import index as jidx
from jax.lax import fori_loop as foril

from qdpy_jax import jax_functions as jf

def f_li_lj_s(Tsr, wsr, r):
    '''Here Tsr is a single value which 
    is a function of l1, l2 and s. 
    wsr is a 1D array in radius.
    '''
    integrand = Tsr * wsr / r

    #### TO BE REPLACED WITH SIMPSON #####
    postintegral = jnp.trapz(integrand, axis=0, x=r)
        
    return postintegral

def build_non_m_part(cenmult_ind, hm_dict, wsr, r, len_s):
    # the total dimension of the hypermatrix
    dim_hyper = hm_dict.dim_hyper
    # the non-m part of the hypermatrix
    non_m_hypmat_s = jnp.zeros((len_s, dim_hyper, dim_hyper))

    # to keep count of the upper tringulr submat index for 
    # correctly indexing the precomputed non-m part
    local_count = 0

    def foril_in_s(s_ind, func_in_s_args):
        non_m_hypmat_in_s, mask_2D, local_count_s = func_in_s_args
        
        # finding the correct index for precomputed part of non-m
        f_li_lj_s_ind = local_count_s +\
                            hm_dict.nbs_cmlcount[cenmult_ind]
        
        # extracting the precomputed non-m part
        GamOmWigTsr = hm_dict.GamOmWigTsr[s_ind, f_li_lj_s_ind]
        
        # computing the constant in that submatrix
        submat_const = f_li_lj_s(GamOmWigTsr,
                                 wsr[s_ind], r)
        
        add_2_hypmat_s = submat_const * mask_2D
        
        non_m_hypmat_in_s = jidx_add(non_m_hypmat_in_s,
                                     jidx[s_ind, :, :],
                                     add_2_hypmat_s)
        
        return non_m_hypmat_in_s, mask_2D, local_count_s

    def fill_non_m_part_1(i, func1_args):
        non_m_hypmat1, local_count1 = func1_args

        # creating the mask for the submatrix
        mask_submat = jnp.outer(hm_dict.trace_arr[i],
                                hm_dict.trace_arr[i])

        # looping over s-summation for diagonal block
        non_m_hypmat1, __, __ = foril(0, len_s, foril_in_s,
                                      (non_m_hypmat1,
                                       mask_submat,
                                       local_count1))

        # incrementing local submat block index after filling
        local_count1 += 1

        
        # filling in the off-diagonal blocks
        def fill_non_m_part_2(j, func2_args):
            non_m_hypmat2, local_count2 = func2_args
            
            # creating the mask for the submatrix                           
            mask_submat = jnp.outer(hm_dict.trace_arr[i],
                                    hm_dict.trace_arr[j])
            
            # looping over s-summation for off-diagonal block           
            non_m_hypmat2, __, __ = foril(0, len_s, foril_in_s,
                                          (non_m_hypmat2,
                                           mask_submat,
                                           local_count2))

            # incrementing local submat block index after filling
            local_count2 += 1

            # the transposed counterpart is handled
            # by multiplying only the UT of m-dependenet part
            # and then doing (HM + HM.CT)/2

            return non_m_hypmat2, local_count2
            
        # loop over off-diagonal blocks for one row
        non_m_hypmat1, local_count1 =\
                foril(i+1,
                      hm_dict.nb_end_ind_arr[cenmult_ind],
                      fill_non_m_part_2, (non_m_hypmat1,local_count1))
        return non_m_hypmat1, local_count1
            
    # loop over diagonal blocks
    non_m_hypmat_s, local_count =\
                    foril(hm_dict.nb_start_ind_arr[cenmult_ind],
                          hm_dict.nb_end_ind_arr[cenmult_ind],
                          fill_non_m_part_1, (non_m_hypmat_s,local_count))
    
    return non_m_hypmat_s


def build_full_hypmat(cenmult_ind, hm_dict, wsr, r, len_s):
    # changes with each Bayesian iteration 
    non_m_hypmat = build_non_m_part(cenmult_ind,
                                    hm_dict,
                                    wsr, r, len_s)

    # initialized with the the first s
    hypmat_full = non_m_hypmat[0] * hm_dict.m_part[0, cenmult_ind]

    def func_add_HM_s(s_ind, hypmat_loc):
        hypmat_loc = jidx_add(hypmat_loc,
                              jidx[:, :],
                              non_m_hypmat[s_ind] *\
                              hm_dict.m_part[s_ind, cenmult_ind])
        return hypmat_loc
        
    # making a loop over s so that it is more generalized
    # starting from index 1 since 0 is already done
    hypmat_full = foril(1, len_s, func_add_HM_s, hypmat_full)

    return hypmat_full

