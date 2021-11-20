import jax.numpy as jnp
from jax.lax import fori_loop as foril
import jax

jidx = jax.ops.index
jidx_update = jax.ops.index_update
jidx_add = jax.ops.index_add

def build_hypmat_w_c(c_dict, hypmat_dict, nc, len_s):
    noc_hypmat = hypmat_dict['noc']
    fix_hypmat = hypmat_dict['fixed']

    # initializing the hypmat
    diag_cs_summed = 0.0*noc_diag['c0_0']

    for s_ind in range(len_s):
        for c_ind in range(nc):
            argstr = f"c{iess}-{inc}"
            diag_cs_summed += c_dict[argstr] * noc_hypmat[argstr]
            
    diag_cs_summed += fix_hypmat
    return diag_cs_summed

    '''
    # looping and summing over s
    def foril_in_s(s_ind, hypmat_s_summed):
        def foril_in_c(c_ind, hypmat_c_summed):
            jidx_add(hypmat_c_summed, jidx[:, :],
                     c_arr[s_ind, c_ind] * noc_hypmat[s_ind][c_ind].todense())
            return hypmat_c_summed

        hypmat_s_summed = foril(0, nc, foril_in_c, hypmat_s_summed)
        return hypmat_s_summed

    hypmat_cs_summed = foril(0, len_s, foril_in_s, hypmat_cs_summed)
    '''
