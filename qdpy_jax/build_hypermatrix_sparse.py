import jax.numpy as jnp
from jax.lax import fori_loop as foril

def build_hypmat_w_c(noc_hypmat, c_arr, nc, len_s):
    '''Function that computes the full
    hypermatrix from the non-c part and
    the c vector. This is for a particular
    cenmult.

    Parameters
    ----------
    noc_hypmat : list of sparse matrices. It is of 
                 shape (s x (dim_hyper x dim_hyper))
                 where the inner bracket shows matrix.

    c_arr : float, array-like
            This is the (s x n_ctrl_pts) matrix
            of control points sampled from Nympyro.
    '''
    hypmat_cs_summed = noc_hypmat[0][0]
    # making it all zero
    hypmat_cs_summed *= 0.0
    
    print(hypmat_cs_summed, noc_hypmat[0][0])

    '''
    # looping and summing over s
    def foril_in_s(s_ind, hypmat_s_summed):
        def foril_in_c(c_ind, hypmat_c_summed):
            print(hypmat_c_summed, noc_hypmat[s_ind][c_ind])
            hypmat_s_summed_out =\
                hypmat_c_summed +\
                (c_arr[s_ind][c_ind] * noc_hypmat[s_ind][c_ind])

            return hypmat_c_summed

        hypmat_s_summed = foril(0, nc, foril_in_c, hypmat_s_summed)
                 
        return hypmat_s_summed

    hypmat_cs_summed = foril(0, len_s, foril_in_s, hypmat_cs_summed)
    '''
    for s_ind in range(len_s):
        for c_ind in range(nc):
            hypmat_cs_summed += c_arr[s_ind][c_ind] * noc_hypmat[s_ind][c_ind]

    return hypmat_cs_summed
        
