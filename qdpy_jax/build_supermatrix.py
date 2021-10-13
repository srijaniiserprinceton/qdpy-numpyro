import numpy as np
import jax.numpy as jnp
import jax
import os
import jax.numpy as jnp
from jax.lax import fori_loop as foril
from collections import namedtuple
from functools import partial

from qdpy_jax import gnool_jit as gjit
from qdpy_jax import class_Cvec as cvec

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"
eig_dir = f"/mnt/disk2/samarth/get-solar-eigs/efs_Jesper/snrnmais_files/eig_files"
'''


def build_SUBMAT_INDICES(CNM_AND_NBS):
    """Returns the namedtuple containing the tiling information
    of the submatrices inside the supermatrix.
    """
    # dim_blocks = np.size(CNM_AND_NBS.omega_nbs)
    # supermatix can be tiled with submatrices corresponding to
    # (l, n) - (l', n') coupling. The dimensions of the submatrix
    # is (2l+1, 2l'+1)
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
    

    # creating the startx, startx, endx, endy for submatrices
    submat_tile_ind = np.zeros((CNM_AND_NBS.dim_blocks, CNM_AND_NBS.dim_blocks, 4), dtype='int32')

    def update_submat_ind_ix(ix, submat_tile_ind):
        def update_submat_ind_iy(iy, submat_tile_ind):
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 0],
                                                   sum_dim[iy, 0])
                                                    
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 2],
                                                   sum_dim[iy, 2])
            
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix, iy, 1],\
                                                   sum_dim[iy, 1])
            
            submat_tile_ind =  jax.ops.index_update(submat_tile_ind,
                                                    jax.ops.index[ix, iy, 3],\
                                                    sum_dim[iy, 3])
            
            return submat_tile_ind

        submat_tile_int = jax.lax.fori_loop(0, CNM_AND_NBS.dim_blocks,
                                            update_submat_ind_iy, submat_tile_ind)
        
        return submat_tile_ind
    

    submat_tile_ind = jax.lax.fori_loop(0, CNM_AND_NBS.dim_blocks,
                                        update_submat_ind_ix, submat_tile_ind)


    # return submat_tile_ind
    # defining the namedtuple
    SUBMAT_DICT_ = namedtuple('SUBMAT_DICT', ['startx',
                                              'starty',
                                              'endx',
                                              'endy']) 

    SUBMAT_DICT = SUBMAT_DICT_(submat_tile_ind[:, :, 0],
                               submat_tile_ind[:, :, 1],
                               submat_tile_ind[:, :, 2],
                               submat_tile_ind[:, :, 3]) 

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
        def build_supermatrix(CNM_AND_NBS, SUBMAT_DICT):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """

            # building the submatrix dictionary
            # SUBMAT_DICT = self.build_SUBMAT_DICT(CNM_AND_NBS)

            # tiling supermatrix with submatrices
            supmat = self.tile_submatrices(CNM_AND_NBS, SUBMAT_DICT)
            return supmat
        return build_supermatrix

    def tile_submatrices(self, GVAR_TR, GVAR_ST, CNM_AND_NBS, SUBMAT_DICT):
        """Function to loop over the submatrix blocks and tile in the
        submatrices into the supermatrix.
        """

        supmat = jnp.zeros((CNM_AND_NBS.dim_super,
                            CNM_AND_NBS.dim_super), dtype='float32')


        # [1, 3, ..., smax]
        s_arr = jnp.arange(1,GVAR_ST.smax+1,2)
    
        # finding omegaref. This is the frequency of the central mode
        # our sorting puts the central mode at the first index in nl_neighbours
        omegaref = CNM_AND_NBS.omega_nbs[0]

        for i in range(CNM_AND_NBS.dim_blocks):
            for ii in range(i, CNM_AND_NBS.dim_blocks):
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                idx1 = CNM_AND_NBS.nl_nbs_idx[i]
                idx2 = CNM_AND_NBS.nl_nbs_idx[ii]

                U1 = np.loadtxt(f'{eig_dir}/U{idx1}.dat')
                V1 = np.loadtxt(f'{eig_dir}/V{idx1}.dat')

                U1 = U1[rmin_ind:rmax_ind]
                V1 = V1[rmin_ind:rmax_ind]
                U2, V2 = U1, V1

                r = jnp.array(r)
                U1, V1 = jnp.array(U1), jnp.array(V1)
                U2, V2 = jnp.array(U2), jnp.array(V2)

                ell1 = CNM_AND_NBS.nl_nbs[0, 1]
                ell2 = CNM_AND_NBS.nl_nbs[1, 1]

                # creating the named tuples
                GVAR = namedtuple('GVAR', 'r wsr s_arr')
                QDPT_MODE = namedtuple('QDPT_MODE', 'ell1 ell2 omegaref')
                EIGFUNCS = namedtuple('EIGFUNCS', 'U1 U2 V1 V2')

                # initializing namedtuples. This could be done from a separate file later
                gvars = GVAR(r, wsr, s_arr)
                qdpt_mode = QDPT_MODE(ell1, ell2, omegaref)
                eigfuncs = EIGFUNCS(U1, U2, V1, V2)

                get_submat = cvec.compute_submatrix(gvars)
                submatdiag = get_submat.jax_get_Cvec()(qdpt_mode, eigfuncs)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                startx, starty = SUBMAT_DICT.startx[i,ii], SUBMAT_DICT.starty[i,ii]
                endx, endy = SUBMAT_DICT.endx[i,ii], SUBMAT_DICT.endy[i,ii]

                # creating the submatrix
                submat = jnp.ones((endx-startx, endy-starty), dtype='float32')
                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[startx:endx,
                                                            starty:endy],
                                              submat)
                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[:100, :100],
                                              jnp.diag(submatdiag[:100]))


                # to avoid repeated filling of the central blocks
                supmat = jax.lax.cond(abs(i-ii)>0,\
                                      lambda __: jax.ops.index_update(supmat,
                                        jax.ops.index[starty:endy, startx:endx],
                                        jnp.transpose(jnp.conjugate(submat))),
                                        lambda __: supmat, operand=None)
        return supmat

'''
class build_supermatrix_functions:
    """Function that returns the function to calculate
    the superMatrix prior to solving eigenvalue problem.
    This function is spefic to the central multiplet. So,
    CENMULT needs to be a static argument.
    """
    def __init__(self):
        pass

    def get_func2build_supermatrix(self):
        def build_supermatrix(CNM_AND_NBS, SUBMAT_DICT):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """

            # building the submatrix dictionary
            # SUBMAT_DICT = self.build_SUBMAT_DICT(CNM_AND_NBS)

            # tiling supermatrix with submatrices
            supmat = self.tile_submatrices(CNM_AND_NBS, SUBMAT_DICT)
            return supmat
        return build_supermatrix

    def tile_submatrices(self, CNM_AND_NBS, SUBMAT_DICT):
        """Function to loop over the submatrix blocks and tile in the
        submatrices into the supermatrix.
        """

        supmat = jnp.zeros((CNM_AND_NBS.dim_super,
                            CNM_AND_NBS.dim_super), dtype='float32')


        s_arr = jnp.array([1,3,5], dtype='int32')

        rmin = 0.3
        rmax = 1.0

        r = np.loadtxt(f'{data_dir}/r.dat') # the radial grid

        # finding the indices for rmin and rmax
        rmin_ind = np.argmin(np.abs(r - rmin))
        rmax_ind = np.argmin(np.abs(r - rmax)) + 1

        # clipping radial grid
        r = r[rmin_ind:rmax_ind]

        # the rotation profile
        wsr = np.loadtxt(f'{data_dir}/w.dat')
        wsr = wsr[:,rmin_ind:rmax_ind]
        wsr = jnp.array(wsr)   # converting to device array once

        # finding omegaref
        omegaref = 1

        for i in range(CNM_AND_NBS.dim_blocks):
            for ii in range(i, CNM_AND_NBS.dim_blocks):
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                idx1 = CNM_AND_NBS.nl_nbs_idx[i]
                idx2 = CNM_AND_NBS.nl_nbs_idx[ii]

                U1 = np.loadtxt(f'{eig_dir}/U{idx1}.dat')
                V1 = np.loadtxt(f'{eig_dir}/V{idx1}.dat')

                U1 = U1[rmin_ind:rmax_ind]
                V1 = V1[rmin_ind:rmax_ind]
                U2, V2 = U1, V1

                r = jnp.array(r)
                U1, V1 = jnp.array(U1), jnp.array(V1)
                U2, V2 = jnp.array(U2), jnp.array(V2)

                ell1 = CNM_AND_NBS.nl_nbs[0, 1]
                ell2 = CNM_AND_NBS.nl_nbs[1, 1]

                # creating the named tuples
                GVAR = namedtuple('GVAR', 'r wsr s_arr')
                QDPT_MODE = namedtuple('QDPT_MODE', 'ell1 ell2 omegaref')
                EIGFUNCS = namedtuple('EIGFUNCS', 'U1 U2 V1 V2')

                # initializing namedtuples. This could be done from a separate file later
                gvars = GVAR(r, wsr, s_arr)
                qdpt_mode = QDPT_MODE(ell1, ell2, omegaref)
                eigfuncs = EIGFUNCS(U1, U2, V1, V2)

                get_submat = cvec.compute_submatrix(gvars)
                submatdiag = get_submat.jax_get_Cvec()(qdpt_mode, eigfuncs)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                startx, starty = SUBMAT_DICT.startx[i,ii], SUBMAT_DICT.starty[i,ii]
                endx, endy = SUBMAT_DICT.endx[i,ii], SUBMAT_DICT.endy[i,ii]

                # creating the submatrix
                submat = jnp.ones((endx-startx, endy-starty), dtype='float32')
                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[startx:endx,
                                                            starty:endy],
                                              submat)
                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[:100, :100],
                                              jnp.diag(submatdiag[:100]))


                # to avoid repeated filling of the central blocks
                supmat = jax.lax.cond(abs(i-ii)>0,\
                                      lambda __: jax.ops.index_update(supmat,
                                        jax.ops.index[starty:endy, startx:endx],
                                        jnp.transpose(jnp.conjugate(submat))),
                                        lambda __: supmat, operand=None)
        return supmat
'''
