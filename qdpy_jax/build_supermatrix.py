import numpy as np
import jax.numpy as jnp
import jax
from jax.lax import fori_loop as foril
from collections import namedtuple

def build_SUBMAT_INDICES(CNM_AND_NBS):
    """Returns the namedtuple containing the tiling information
    of the submatrices inside the supermatrix.
    """
    # dim_blocks = np.size(CNM_AND_NBS.omega_nbs)
    # supermatix can be tiled with submatrices corresponding to
    # (l, n) - (l', n') coupling. The dimensions of the submatrix
    # is (2l+1, 2l'+1)
    dim_blocks = CNM_AND_NBS.dim_blocks

    dimX_submat = 2*CNM_AND_NBS.nl_nbs[:, 1].reshape(1, dim_blocks) \
            * np.ones((dim_blocks, 1), dtype='int32') + 1
    dimY_submat = 2*CNM_AND_NBS.nl_nbs[:, 1].reshape(dim_blocks, 1) \
            * np.ones((1, dim_blocks), dtype='int32') + 1

    sum_dim = np.zeros((dim_blocks, 4), dtype='int32')

    for i in range(dim_blocks):
        sum_dim[i, 0] = int(dimX_submat[0, :i].sum())
        sum_dim[i, 1] = int(dimY_submat[:i, 0].sum())
        sum_dim[i, 2] = int(dimX_submat[0, :int(i+1)].sum())
        sum_dim[i, 3] = int(dimY_submat[:int(i+1), 0].sum())

    sum_dim = jnp.asarray(sum_dim)

    # creating the startx, startx, endx, endy for submatrices
    submat_tile_ind = np.zeros((dim_blocks, dim_blocks, 4), dtype='int32')
    def update_submat_ind_i(i, submat_tile_ind):
        submat_tile_ind = foril(0, CNM_AND_NBS.dim_blocks,
                                lambda ii, submat_tile_ind:\
                                jax.ops.index_update(submat_tile_ind,
                                                     jax.ops.index[i, ii, 0],
                                                     sum_dim[ii, 0]),
                                submat_tile_ind)
        
        submat_tile_ind = foril(0, CNM_AND_NBS.dim_blocks,
                                lambda ii, submat_tile_ind:\
                                jax.ops.index_update(submat_tile_ind,
                                                     jax.ops.index[i, ii, 2],
                                                     sum_dim[ii, 2]),
                                submat_tile_ind)

        submat_tile_ind = \
            foril(0, CNM_AND_NBS.dim_blocks,
                  lambda ii, submat_tile_ind:\
                  jax.ops.index_update(submat_tile_ind,
                                       jax.ops.index[i, ii, 1],\
                                       sum_dim[ii, 1]),
                  submat_tile_ind)

        submat_tile_ind = \
            foril(0, CNM_AND_NBS.dim_blocks,
                  lambda ii, submat_tile_ind:\
                  jax.ops.index_update(submat_tile_ind,
                                       jax.ops.index[i, ii, 3],\
                                       sum_dim[ii, 3]),
                  submat_tile_ind)
        return submat_tile_ind

    submat_tile_ind = jax.lax.fori_loop(0, CNM_AND_NBS.dim_blocks,
                                        update_submat_ind_i, submat_tile_ind)

    '''
    for ix in range(CNM_AND_NBS.dim_blocks):
        for iy in range(CNM_AND_NBS.dim_blocks):
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix,iy,0],
                                                   int(dimX_submat[0, :ix].sum()))
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix,iy,1],
                                                   int(dimY_submat[:iy, 0].sum()))
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix,iy,2],
                                                   int(dimX_submat[0, :int(ix+1)].sum()))
            submat_tile_ind = jax.ops.index_update(submat_tile_ind,
                                                   jax.ops.index[ix,iy,3],
                                                   int(dimY_submat[:int(iy+1), 0].sum()))
    '''


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

    def tile_submatrices(self, CNM_AND_NBS, SUBMAT_DICT):
        """Function to loop over the submatrix blocks and tile in the
        submatrices into the supermatrix.
        """

        supmat = jnp.zeros((CNM_AND_NBS.dim_super,
                            CNM_AND_NBS.dim_super), dtype='float32')

        for i in range(CNM_AND_NBS.dim_blocks):
            for ii in range(i, CNM_AND_NBS.dim_blocks):
                startx, starty = SUBMAT_DICT.startx[i,ii], SUBMAT_DICT.starty[i,ii]
                endx, endy = SUBMAT_DICT.endx[i,ii], SUBMAT_DICT.endy[i,ii]

                # creating the submatrix
                submat = jnp.ones((endx-startx, endy-starty), dtype='float32')
                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[startx:endx,
                                                            starty:endy],
                                              submat)

                # to avoid repeated filling of the central blocks
                supmat = jax.lax.cond(abs(i-ii)>0,\
                                      lambda __: jax.ops.index_update(supmat,
                                        jax.ops.index[starty:endy, startx:endx],
                                        jnp.transpose(jnp.conjugate(submat))), 
                                        lambda __: supmat, operand=None)
        return supmat      
