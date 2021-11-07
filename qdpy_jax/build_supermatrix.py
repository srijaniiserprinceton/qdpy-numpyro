import numpy as np
import jax.numpy as jnp
import jax
import sys
import os
from jax.lax import fori_loop as foril
from collections import namedtuple
from functools import partial
from jax import tree_util as tu

from qdpy_jax import class_Cvec as cvec
from qdpy_jax import jax_functions as jf

jidx = jax.ops.index
jidx_update = jax.ops.index_update


def eigval_sort_slice(eigval, eigvec):
    def body_func(i, ebs):
        return jidx_update(ebs, jidx[i], jnp.argmax(jnp.abs(eigvec[i])))

    eigbasis_sort = np.zeros(len(eigval), dtype=int)
    eigbasis_sort = foril(0, len(eigval), body_func, eigbasis_sort)

    return eigval[eigbasis_sort]


def build_SUBMAT_INDICES(CNM_AND_NBS):
    # supermatix can be tiled with submatrices corresponding to
    # (l, n) - (l', n') coupling. The dimensions of the submatrix
    # is (2l+1, 2l'+1)
    dim_blocks = len(CNM_AND_NBS.omega_nbs)
    # nl array of neighbours
    nl_nbs = np.asarray(CNM_AND_NBS.nl_nbs)

    dimX_submat = 2 * nl_nbs[:, 1].reshape(1, dim_blocks) \
                  * np.ones((dim_blocks, 1), dtype='int32') + 1
    dimY_submat = dimX_submat.T 

    # creating the startx, startx, endx, endy for submatrices
    submat_tile_ind = np.zeros((dim_blocks,
                                dim_blocks, 4), dtype='int32') 

    for ix in range(0, dim_blocks):
        for iy in range(0, dim_blocks):
            submat_tile_ind[ix, iy, 0] = np.sum(dimX_submat[0, :ix])

            submat_tile_ind[ix, iy, 1] = np.sum(dimY_submat[:iy, 0])
                                         
            submat_tile_ind[ix, iy, 2] = np.sum(dimX_submat[0, :ix+1])
        
            submat_tile_ind[ix, iy, 3] = np.sum(dimY_submat[:iy+1, 0])

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

class build_supermatrix_functions:
    """Function that returns the function to calculate
    the superMatrix prior to solving eigenvalue problem.
    This function is spefic to the central multiplet. So,
    CENMULT needs to be a static argument.
    """
    def __init__(self):
        pass

    def get_func2build_supermatrix(self):
        # @partial(jax.jit, static_argnums=(0, 1, 2))
        def build_supermatrix(CNM_AND_NBS, SUBMAT_DICT, GVARS_ST, GVARS_TR, ctrl_arr):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """
            # tiling supermatrix with submatrices
            supmat = self.tile_submatrices(CNM_AND_NBS, SUBMAT_DICT,
                                           GVARS_ST, GVARS_TR, ctrl_arr)
            return supmat
        return build_supermatrix

    def tile_submatrices(self, CNM_AND_NBS, SUBMAT_DICT, GVARS_ST, GVARS_TR, ctrl_arr):
        """Function to loop over the submatrix blocks and tile in the
        submatrices into the supermatrix.
        """

        # our sorting puts the central mode at the first index in nl_neighbours
        omegaref = CNM_AND_NBS.omega_nbs[0]

        # changing required tuples to arrays and lists
        nl_idx_pruned = list(GVARS_ST.nl_idx_pruned)
        nl_nbs = np.asarray(CNM_AND_NBS.nl_nbs)

        dim_super = np.sum(2*nl_nbs[:, 1] + 1)
        dim_blocks = len(CNM_AND_NBS.omega_nbs)
        supmat = jnp.zeros((dim_super, dim_super))
        
        # finding omegaref. This is the frequency of the central mode
        startx_arr = np.asarray(SUBMAT_DICT.startx)
        starty_arr = np.asarray(SUBMAT_DICT.starty)
        endx_arr = np.asarray(SUBMAT_DICT.endx)
        endy_arr = np.asarray(SUBMAT_DICT.endy)
        
        for ic in range(dim_blocks):
            for ir in range(ic, dim_blocks):
                idx1 = CNM_AND_NBS.nl_nbs_idx[ir]
                idx2 = CNM_AND_NBS.nl_nbs_idx[ic]

                idx1 = nl_idx_pruned.index(CNM_AND_NBS.nl_nbs_idx[ir])
                idx2 = nl_idx_pruned.index(CNM_AND_NBS.nl_nbs_idx[ic])

                U1, V1 = GVARS_TR.U_arr[idx1], GVARS_TR.V_arr[idx1]
                U2, V2 = GVARS_TR.U_arr[idx2], GVARS_TR.V_arr[idx2]

                U1, V1 = jnp.array(U1), jnp.array(V1)
                U2, V2 = jnp.array(U2), jnp.array(V2)

                # because ii starts from i, we only scan
                # the region where ell2 >= ell1
                ell1 = nl_nbs[ir, 1]
                ell2 = nl_nbs[ic, 1]
                
                # creating the named tuples
                gvars = jf.create_namedtuple('GVAR',
                                             ['r',
                                              'r_spline',
                                              'rth_ind',
                                              'wsr',
                                              's_arr',
                                              'knot_arr',
                                              'spl_deg'],
                                             (GVARS_TR.r,
                                              GVARS_TR.r_spline,
                                              GVARS_ST.rth_ind,
                                              GVARS_TR.wsr,
                                              GVARS_ST.s_arr,
                                              GVARS_TR.knot_arr,
                                              GVARS_ST.spl_deg))

                qdpt_mode = jf.create_namedtuple('QDPT_MODE',
                                                 ['ell1',
                                                  'ell2',
                                                  'ellmin',
                                                  'omegaref'],
                                                 (ell1,
                                                  ell2,
                                                  min(ell1, ell2),
                                                  omegaref))

                eigfuncs = jf.create_namedtuple('EIGFUNCS',
                                                ['U1', 'U2',
                                                 'V1', 'V2'],
                                                (U1, U2,
                                                 V1, V2))

                wigs = jf.create_namedtuple('WIGNERS',
                                            ['wig_list',
                                             'wig_idx'],
                                            (GVARS_TR.wig_list,
                                             np.asarray(GVARS_ST.wig_idx)))

                get_submat = cvec.compute_submatrix(gvars)

                submatdiag = get_submat.jax_get_Cvec()(qdpt_mode, eigfuncs, wigs, ctrl_arr)

                startx, starty = startx_arr[ir, ic], starty_arr[ir, ic]
                endx, endy = endx_arr[ir, ic], endy_arr[ir, ic]

                # creating the rectangular submatrix
                submat = jnp.zeros((endx-startx, endy-starty))

                # ell1 goes along the x-axis of the matrix and ell2 goes along the y-axis
                dell = ell1 - ell2
                absdell = np.abs(dell)

                # tiling the diagonal submatdiag into the rect submatrix
                # submat clipping index in the order of x_left, y_up
                sm_clip_ind = np.array([0, 0], dtype='int')

                def dell_not_zero_func(sm_clip_ind):
                    sm_clip_ind = jax.lax.cond(dell > 0,
                                               lambda __: np.array([absdell, 0]),
                                               lambda __: np.array([0, absdell]),
                                               operand = None)
                    return sm_clip_ind

                sm_clip_ind = jax.lax.cond(dell == 0,
                                           lambda sm_clip_ind: sm_clip_ind,
                                           dell_not_zero_func,
                                           operand = sm_clip_ind)
                                        
    
                submat = jax.lax.dynamic_update_slice(submat,
                                                      jnp.diag(submatdiag),
                                                      (sm_clip_ind[0], sm_clip_ind[1]))

                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[startx:endx, starty:endy],
                                              submat)

                # to avoid repeated filling of the central blocks
                supmat = jax.lax.cond(abs(ic-ir)>0,\
                                      lambda __: jax.ops.index_update(supmat,
                                                jax.ops.index[starty:endy, startx:endx],
                                                jnp.transpose(jnp.conjugate(submat))),
                                      lambda __: supmat, operand=None)

            # filling the freqdiag
            omega_nb = CNM_AND_NBS.omega_nbs[ic]
            startx, endx = startx_arr[ic, ic], endx_arr[ic, ic]
            om2diff = omega_nb**2 - omegaref**2
            om2diff_mat = jnp.identity(endx-startx) * om2diff
            supmat = jax.ops.index_add(supmat,
                                       jax.ops.index[startx:endx, startx:endx],
                                       om2diff_mat)

        return supmat


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
