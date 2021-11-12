import jax.numpy as jnp
from jax.ops import index_add as jidx_add
from jax.ops import index as jidx
from jax.lax import fori_loop as foril

def f_li_lj_s():
    return 1.0

def build_non_m_uppertriang(cenmult_ind, hm_dict):
    # the total dimension of the hypermatrix
    dim_hyper = hm_dict.dim_hyper
    # the non-m part of the hypermatrix
    non_m_hypmat = jnp.zeros((dim_hyper, dim_hyper))

    # fillin in the non-m part using the masks
    def fill_non_m_part_1(i, non_m_hypmat1):
        # filling in the diagonal blocks
        submat_const = f_li_lj_s()
        add_2_hypmat1 = submat_const *\
                       jnp.outer(hm_dict.trace_arr[i],
                                 hm_dict.trace_arr[i])

        non_m_hypmat1 = jidx_add(non_m_hypmat1,
                                 jidx[:,:],
                                 add_2_hypmat1)
        
        # filling in the off-diagonal blocks
        def fill_non_m_part_2(j, non_m_hypmat2):
            submat_const = f_li_lj_s()
            add_2_hypmat = submat_const *\
                           jnp.outer(hm_dict.trace_arr[i],
                                     hm_dict.trace_arr[j])

            non_m_hypmat2 = jidx_add(non_m_hypmat2,
                                     jidx[:,:],
                                     add_2_hypmat)
            return non_m_hypmat2

        non_m_hypmat1 = foril(i+1,
                              hm_dict.nb_end_ind_arr[cenmult_ind],
                              fill_non_m_part_2, non_m_hypmat1)
        return non_m_hypmat1
    
    non_m_hypmat = foril(hm_dict.nb_start_ind_arr[cenmult_ind],
                         hm_dict.nb_end_ind_arr[cenmult_ind],
                         fill_non_m_part_1, non_m_hypmat)

    return non_m_hypmat


def build_full_hypmat(cenmult_ind, hm_dict, wig_hyper):
    non_m_UT = build_non_m_uppertriang(cenmult_ind, hm_dict)
    hypmat_UT = non_m_UT * wig_hyper[0, ...]
    hypmat_UT = jidx_add(hypmat_UT, jidx[:, :], non_m_UT * wig_hyper[1, ...])
    hypmat_UT = jidx_add(hypmat_UT, jidx[:, :], non_m_UT * wig_hyper[2, ...])
    hypmat_full = (hypmat_UT + hypmat_UT.T.conj())
    return hypmat_full

