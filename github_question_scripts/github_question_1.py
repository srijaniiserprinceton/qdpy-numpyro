import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
import sys

class qdptMode:
    def __init__(self, GVAR):
        """Initialized with parameters that do not change
        with the central multiplet CENMULT.
        """
        self.smax = GVAR.smax

    def get_mode_neighbors_params(self, CENMULT, NBR_DICT):
        """Updates the parameters corresponding to mode-neighbours.
        CENMULT is passed as a static variable since this update needs
        to happen only when CENMULT changes.
        """
        # THESE PARAMETERS OUGHT TO BE CONSTRUCTED ON THE FLY FROM CENMULT
        nl_neighbours = NBR_DICT.nl_neighbours + 1
        omega_neighbours = NBR_DICT.omega_neighbours + 1
        num_neighbours = jnp.size(NBR_DICT.omega_neighbours)

        NBR_DICT_FINAL = namedtuple('NBR_DICT_FINAL',
                                    'nl_neighbours \
                                    omega_neighbours \
                                    num_neighbours')
        
        NBR_DICT_FINAL = NBR_DICT_FINAL(nl_neighbours,
                                    omega_neighbours,
                                    num_neighbours)

        

        return NBR_DICT_FINAL
    

    def build_supermatrix_function(self):
        """Function that returns the function to calculate
        the superMatrix prior to solving eigenvalue problem. 
        This function is specific to the central multiplet. So,
        CENMULT needs to be a static argument.
        """
        def compute_supermatrix(CENMULT, NBR_DICT_INIT):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """
            # getting the neighbour dictionary from CENMULT
            NBR_DICT = self.get_mode_neighbors_params(CENMULT, NBR_DICT_INIT)

            # unpacking the neighbour dictionary
            # number of submatrices along each dimension of the square supermatrix
            dim_blocks = NBR_DICT.num_neighbours

            nl_neighbours = NBR_DICT.nl_neighbours
            
            # supermatix can be tiled with submatrices corresponding to                                                                                                      
            # (l, n) - (l', n') coupling. The dimensions of the submatrix                                                                                                    
            # is (2l+1, 2l'+1)                                                                                                                                               
            dimX_submat = 2*nl_neighbours[:, 1].reshape(1, dim_blocks) \
                          * jnp.ones((dim_blocks, 1), dtype='int32') + 1
            dimY_submat = 2*nl_neighbours[:, 1].reshape(dim_blocks, 1) \
                          * jnp.ones((1, dim_blocks), dtype='int32') + 1

            # we use float32 for the current problem since for DR it is a real matrix
            supmat = jnp.zeros((CENMULT.dim_super, CENMULT.dim_super), dtype='float32')
            
            return supmat
            
        return compute_supermatrix

# initializing the necessary variables in gvar
fwindow = 150   # in muHz
# central multiplet
ell0 = 200
# max degree of perturbation (considering odd only)
smax = 5

# loading the neighbours. Here, we have hardcoded the following multiplets
# (0,198), (0,200), (0,202). 
nl_neighbours = np.array([[0,198], [0,200], [0,202]], dtype='int32')
omega_neighbours = np.array([67.65916460412984, 67.99455100807411, 68.32826640883721])

# calculating the dimension of the supermatrix beforehand
dim_super = np.sum(2 * nl_neighbours[:,1] + 1)

# converting arrays to jax.numpy type
nl_neighbours = jnp.array(nl_neighbours)
omega_neighbours = jnp.array(omega_neighbours)

# this dictionary does not change with changing central multiplets
GVAR_ = namedtuple('GVAR', 'fwindow smax')

# this dictionary changes with central multiplet
# NEEDS TO BE A STATIC ARGUMENT
CENMULT_ = namedtuple('CENMULT', 'ell0, dim_super')

# dictionary for neighbours
NEIGHBOUR_DICT_ = namedtuple('NEIGHBOUR_DICT', 'nl_neighbours omega_neighbours')

GVAR = GVAR_(fwindow, smax)
CENMULT = CENMULT_(ell0, dim_super)
NEIGHBOUR_DICT = NEIGHBOUR_DICT_(nl_neighbours, omega_neighbours)
# creating instance of the class qdptMode
qdpt_mode = qdptMode(GVAR)

# jitting function to compute supermatrix
_compute_supermatrix = jax.jit(qdpt_mode.build_supermatrix_function(), static_argnums=(0,))

__ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT).block_until_ready()

# checking if its re-compiling
for ell0 in range(190, 201):
    print('Executing ', ell0)
    CENMULT = CENMULT_(ell0, dim_super)
    __ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT).block_until_ready()


# checking if its re-compiling
for ell0 in range(180, 211):
    print('Executing ', ell0)
    CENMULT = CENMULT_(ell0, dim_super)
    __ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT).block_until_ready()