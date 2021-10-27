from collections import namedtuple
from jax.lax import fori_loop as foril
from functools import partial
import jax.numpy as jnp
import numpy as np
import py3nj
import time
import sys

import jax
import jax.tree_util as tu

# new package in jax.numpy
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import load_multiplets
from qdpy_jax import jax_functions as jf
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import prune_multiplets

jax.config.update('jax_platform_name', 'cpu')




def build_SUBMAT_INDICES(CNM_AND_NBS):
    """Returns the namedtuple containing the tiling information
    of the submatrices inside the supermatrix.
    """
    print(f"Entering build_SUBMAT_INDICES")
    dim_blocks = CNM_AND_NBS.dim_blocks
    dimX_submat = 2*CNM_AND_NBS.nl_nbs[:, 1].reshape(1, CNM_AND_NBS.dim_blocks) \
            * np.ones((CNM_AND_NBS.dim_blocks, 1), dtype='int32') + 1
    dimY_submat = dimX_submat.T
    sum_dim = jnp.zeros((CNM_AND_NBS.dim_blocks, 4), dtype='int32')

    for i in range(CNM_AND_NBS.dim_blocks):
        sum_dim = jax.ops.index_update(sum_dim,
                                       jax.ops.index[i, 0],
                                       jnp.sum(dimX_submat[0, :i]))
        sum_dim = jax.ops.index_update(sum_dim,
                                       jax.ops.index[i, 1],
                                       jnp.sum(dimY_submat[:i, 0]))
        sum_dim = jax.ops.index_update(sum_dim,
                                       jax.ops.index[i, 2],
                                       jnp.sum(dimX_submat[0, :i+1]))
        sum_dim = jax.ops.index_update(sum_dim,
                                       jax.ops.index[i, 3],
                                       jnp.sum(dimY_submat[:i+1, 0]))
    print(f" -- Computing sum_dim")

    # creating the startx, startx, endx, endy for submatrices
    submat_tile_ind = np.zeros((CNM_AND_NBS.dim_blocks,
                                CNM_AND_NBS.dim_blocks, 4), dtype='int32')


    def update_submat_ind_ix(ix, submat_tile_ind):
        def update_submat_ind_iy(iy, submat_tile_ind):
            print(f" -- Entering iy loop {ix} {iy}")
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 0],
                                                   5)
                                                   # sum_dim[ix, 0])
                                                    
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 1],
                                                   5)
                                                   # sum_dim[iy, 1])
            
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 2],
                                                   5)
                                                   # sum_dim[ix, 2])
            
            submat_tile_ind =  jax.ops.index_update(submat_tile_ind,
                                                    jax.ops.index[ix, iy, 3],
                                                    5)
                                                    # sum_dim[iy, 3])
            return submat_tile_ind

        print(f" -- Entering ix loop")
        submat_tile_int = foril(0, dim_blocks,
                                update_submat_ind_iy,
                                submat_tile_ind)
        return submat_tile_ind

    # print(f" -- dim_blocks = {dim_blocks}")
    submat_tile_ind = foril(0, dim_blocks,
                            update_submat_ind_ix,
                            submat_tile_ind)

    # creating the submat-dictionary namedtuple
    SUBMAT_DICT = jf.create_namedtuple('SUBMAT_DICT',
                                       ['startx',
                                        'starty',
                                        'endx',
                                        'endy'],
                                       (submat_tile_ind[:, :, 0],
                                        submat_tile_ind[:, :, 1],
                                        submat_tile_ind[:, :, 2],
                                        submat_tile_ind[:, :, 3])) 
    return SUBMAT_DICT

build_SUBMAT_INDICES_ = gjit.gnool_jit(build_SUBMAT_INDICES,
                                       static_array_argnums=(0,))

def get_pruned_multiplets(nl, omega, nl_all):
    n1 = nl[:, 0]
    l1 = nl[:, 1]

    omega_pruned = [omega[0]]
    
    nl_idx_pruned = [nl_all.tolist().index([nl[0, 0], nl[0, 1]])]
    nl_pruned = nl[0, :].reshape(1, 2)

    for i in range(1, len(n1)):
        try:
            nl_pruned.tolist().index([n1[i], l1[i]])
        except ValueError:
            nl_pruned = np.concatenate((nl_pruned,
                                        nl[i, :].reshape(1, 2)), 0)
            omega_pruned.append(omega[i])
            nl_idx_pruned.append(nl_all.tolist().index([nl[i, 0], nl[i, 1]]))
    return nl_pruned, nl_idx_pruned, omega_pruned



if __name__ == "__main__":
    GVARS = gvar_jax.GlobalVars()
    GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

    # jitting various functions
    get_namedtuple_for_cenmult_and_neighbours_ = \
        gjit.gnool_jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                    static_array_argnums = (0, 1, 2))

    # extracting the pruned parameters for multiplets of interest
    nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx_full = \
        prune_multiplets.get_pruned_attributes(GVARS, GVARS_ST)

    lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                        nl_idx_pruned,
                                        omega_pruned)

    GVARS_PRUNED_ST = jf.create_namedtuple('GVARS_ST',
                                           ['s_arr',
                                            'nl_all',
                                            'nl_idx_pruned',
                                            'omega_list',
                                            'fwindow',
                                            'OM',
                                            'wig_idx_full'],
                                           (GVARS_ST.s_arr,
                                            lm.nl_pruned,
                                            lm.nl_idx_pruned,
                                            lm.omega_pruned,
                                            GVARS_ST.fwindow,
                                            GVARS_ST.OM,
                                            wig_idx_full))

    nmults = len(GVARS.n0_arr)

    n0, ell0 = GVARS.n0_arr[0], GVARS.ell0_arr[0]
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
    print(SUBMAT_DICT.startx)
