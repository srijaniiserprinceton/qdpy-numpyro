import jax
import jax.numpy as jnp
from collections import namedtuple

class qdptMode:
    """Class that handles modes that are perturbed using QDPT. Each class instance                                                                                           
    corresponds to a central mode (l0, n0). The frequency space is scanned to find out                                                                      
    all the neighbouring modes (l, n) which interact with the central mode                                                                                             
    (and amongnst themselves). The supermatrix is constructed for all possible                                                                                   
    coupling combinations (l, n) <--> (l', n').                                                                                                                      
    """
    __all__ = ["nl_idx", "nl_idx_vec",
               "get_omega_neighbors",
               "get_mode_neighbors_params",
               "create_supermatrix",
               "update_supermatrix"]

    def __init__(self, GVAR):
        """Initialized with parameters that do not change
        with the central multiplet.
        """
        self.smax = GVAR.smax
        self.freq_window = GVAR.fwindow

    def get_mode_neighbors_params(self, CENMULT, NBR_DICT):
        """Gets the parameters corresponding to mode-neighbours.
        CENMULT is passes so as to make that a static variable.
        """
        # THESE PARAMETERS OUT TO BE CONSTRUCTED ON THE FLY FROM
        # CENMULT
        nl_neighbours = NBR_DICT.nl_neighbours
        nl_neighbours_idx = NBR_DICT.nl_neighbours_idx
        omega_neighbours = NBR_DICT.omega_neighbours
        num_neighbours = jnp.size(NBR_DICT.omega_neighbours)

        NBR_DICT_FINAL = namedtuple('NBR_DICT_FINAL',
                                    'nl_neighbours \
                                    nl_neighbours_idx \
                                    omega_neighbours \
                                    num_neighbours')
        
        NBR_DICT_FINAL = NBR_DICT_FINAL(nl_neighbours,
                                    nl_neighbours_idx,
                                    omega_neighbours,
                                    num_neighbours)

        

        return NBR_DICT_FINAL
    

    def build_supermatrix_function(self):
        """Function that returns the function to calculate
        the superMatrix prior to solving eigenvalue problem. 
        This function is spefic to the central multiplet. So,
        CENMULT needs to be a static argument.
        """
        def compute_supermatrix(CENMULT, NBR_DICT_TRIAL):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """
            # getting the neighbour dictionary from CENMULT
            # hardcoding its contnet for now, hence passing 
            # NBR_DICT_TRIAL TOO
            NBR_DICT = self.get_mode_neighbors_params(CENMULT, NBR_DICT_TRIAL)

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
            dim_super = jnp.sum(dimX_submat[0, :])
            # we use float32 for the current problem since for DR it is a real matrix
            supmat = jnp.zeros((dim_super, dim_super), dtype='float32')
            
            return supmat
            
        return compute_supermatrix

# initializing the necessary variables in gvar
fwindow = 150   # in muHz
# central multiplet
n0, ell0 = 0, 200
# max degree of perturbation (considering odd only)
smax = 5
# hardcoding central multiplet index for (0, 200)
cenmult_idx = 3672 
# hardcong unperturbed freq of central multiplet
# needs to be scaled up by GVAR.OM * 1e6 to be in muHz
unit_omega = 20.963670602632025   # GVAR.OM * 1e6  
omegaref = 67.99455100807411 

# loading the neighbours. Here, we have hardcoded the following multiplets
# (0,198), (0,200), (0,202). 
nl_neighbours = jnp.array([[0,198], [0,200], [0,202]], dtype='int32')
nl_neighbours_idx = jnp.array([3650, 3672, 3693], dtype='int32')
omega_neighbours = jnp.array([67.65916460412984, 67.99455100807411, 68.32826640883721])

# this dictionary does not change with changing central multiplets
GVAR = namedtuple('GVAR', 'fwindow smax unit_omega')

# this dictionary changes with central multiplet
# NEEDS TO BE A STATIC ARGUMENT
CENMULT = namedtuple('CENMULT', 'n0 ell0 omegaref cenmult_idx')

# dictionary for neighbours
NEIGHBOUR_DICT = namedtuple('NEIGHBOUR_DICT', 'nl_neighbours nl_neighbours_idx, omega_neighbours')

GVAR = GVAR(fwindow, smax, unit_omega)
CENMULT = CENMULT(n0, ell0, omegaref, cenmult_idx)
NEIGHBOUR_DICT = NEIGHBOUR_DICT(nl_neighbours, nl_neighbours_idx, omega_neighbours)
# creating instance of the class qdptMode
qdpt_mode = qdptMode(GVAR)

# jitting function to compute supermatrix
_compute_supermatrix = jax.jit(qdpt_mode.build_supermatrix_function(), static_argnums=(0,))

__ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT).block_until_ready()
