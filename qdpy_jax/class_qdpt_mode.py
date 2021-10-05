import jax
import jax.numpy as jnp
# we still need numpy for the static variables
# for example, variables that conrtol the dimensions of matrices
import numpy as np   
from collections import namedtuple
from functools import partial
import sys
import time
# imports from qdpy_jax
from qdpy_jax import globalvars
from qdpy_jax import gnool_jit as gjit 

#------((( creating the namedtuples of global variables --------

GVARS = globalvars.GlobalVars()
# the global path variables
GVARS_PATHS = GVARS.get_namedtuple_gvar_paths()
# the global traced variables
GVARS_TR = GVARS.get_namedtuple_gvar_traced()
# the global static variables 
GVARS_ST = GVARS.get_namedtuple_gvar_static()

#------- creating the namedtuples of global variables )))-------

#--------------((( creating qdptMode namedtuple ----------------

def nl_idx(n0, l0):
    try:
        idx = GVARS_TR.nl_all_list.index([n0, l0])
    except ValueError:
        idx = None
        logger.error('Mode not found')
    return idx

def nl_idx_vec(nl_list):
    nlnum = nl_list.shape[0]
    nlidx = np.zeros(nlnum, dtype='int32')
    for i in range(nlnum):
        nlidx[i] = nl_idx(nl_list[i][0],
                          nl_list[i][1])
    return nlidx

def get_omega_neighbors(nl_idx):
    nlnum = len(nl_idx)
    omega_neighbors = np.zeros(nlnum)
    for i in range(nlnum):
        omega_neighbors[i] = GVARS_TR.omega_list[nl_idx[i]]
    return omega_neighbors

def get_namedtuple_for_cenmult_and_neighbours(n0, ell0):
    """Function that returns the name tuple for the
    attributes of the central mode and the neighbours for 
    that central mode. n0 and ell0 are static since everything
    else depends on n0 and ell0."""

    # print(f'Assembling namedtuple for ({n0}, {ell0})')

    # unperturbed frequency of central multiplet (n0, ell0)
    mult_idx = GVARS_TR.nl_all_list.index([n0, ell0])
    omega0 = GVARS_TR.omega_list[mult_idx]

    
    omega_list = GVARS_TR.omega_list
    nl_all = GVARS_TR.nl_all
    omega_diff = (omega_list - omega0) * GVARS_TR.OM * 1e6

    # defining various masks to minimize the multiplet-couplings
 
    # rejecting modes far in frequency
    mask_omega = abs(omega_diff) <= GVARS_TR.fwindow 
    
    # rejecting modes that don't satisfy triangle inequality
    mask_ell = abs(nl_all[:, 1] - ell0) <= GVARS_ST.smax

    # only even l1-l2 is coupled for odd-s rotation perturbation
    # this is specific to the fact that DR is considered for odd s only
    mask_odd = ((GVARS_TR.nl_all[:, 1] - ell0)%2) == 0
    
    # creating the final mask accounting for all of the masks above
    mask_nb = mask_omega * mask_ell * mask_odd

    # sorting the multiplets in ascending order of distance from (n0, ell0)
    sort_idx = np.argsort(abs(omega_diff[mask_nb]))
    
    # the final attributes that will be stored
    nl_neighbours = nl_all[mask_nb][sort_idx]
    nl_neighbours_idx = nl_idx_vec(nl_neighbours)
    omega_neighbours = get_omega_neighbors(nl_neighbours_idx)
    num_neighbours = len(nl_neighbours_idx)
    
    # creating the namedtuple
    CENMULT_AND_NBS_ = namedtuple('CENMULT_AND_NBS', ['nl_nbs',
                                                      'nl_nbs_idx',
                                                      'omega_nbs',
                                                      'num_nbs'])
    
    CENMULT_AND_NBS = CENMULT_AND_NBS_(nl_neighbours,
                                       nl_neighbours_idx,
                                       omega_neighbours,
                                       num_neighbours)
    
    return CENMULT_AND_NBS

# defining the central mode (n0, ell0)
n0 ,ell0 = 1, 150

# obtaining the namedtuple for the central mode and its neighbours
get_namedtuple_for_cenmult_and_neighbours_ = jax.jit(get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))
CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)

'''
# Checking jitting
n0 = 0
t1 = time.time()
for ell in range(195, 290):
    # print(f'Executing {n0}, {ell}')
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell)
t2 = time.time()

print(f'Compiling in: {t2-t1} seconds')

t3 = time.time()
for ell in range(195, 290):
    # print(f'Executing {n0}, {ell}')
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell)
t4 = time.time()

print(f'Compiling in: {t4-t3} seconds.')
'''
#---------creating qdptMode namedtuple )))-----------

class qdptMode:
    """Class that handles modes that are perturbed using QDPT. 
    Each class instance corresponds to a central mode (l0, n0). 
    The frequency space is scanned to find out
    all the neighbouring modes (l, n) which interact 
    with the central mode (and amongnst themselves). 
    The supermatrix is constructed for all possible
    coupling combinations (l, n) <-> (l', n').
    """
    __all__ = ["get_mode_neighbours_params",
               "build_supermatrix_function"]

    def __init__(self, GVAR):
        """Initialized with parameters that do not change
        with the central multiplet.
        """
        self.smax = GVAR.smax
        self.freq_window = GVAR.fwindow

    def get_mode_neighbours_params(self, CENMULT, NBR_DICT):
        """Gets the parameters corresponding to mode-neighbours.
        CENMULT is passes so as to make that a static variable.
        """
        # THESE PARAMETERS OUGHT TO BE CONSTRUCTED ON THE FLY FROM CENMULT
        nl_neighbours = NBR_DICT.nl_neighbours
        nl_neighbours_idx = NBR_DICT.nl_neighbours_idx
        omega_neighbours = NBR_DICT.omega_neighbours

        NBR_DICT_FINAL = namedtuple('NBR_DICT_FINAL',
                                    ['nl_neighbours',
                                    'nl_neighbours_idx',
                                    'omega_neighbours'])
        
        NBR_DICT_FINAL = NBR_DICT_FINAL(nl_neighbours,
                                        nl_neighbours_idx,
                                        omega_neighbours)

        return NBR_DICT_FINAL


    def build_supermatrix_function(self):
        """Function that returns the function to calculate
        the superMatrix prior to solving eigenvalue problem. 
        This function is spefic to the central multiplet. So,
        CENMULT needs to be a static argument.
        """
        def compute_supermatrix(CENMULT, NBR_DICT_TRIAL, SUBMAT_DICT):
            """Function to assimilate all the neighbour info
            and return the function to compute the SuperMatrix'
            """
            # print('Compiling dummy: ', CENMULT.dummy)
            # tiling supermatrix with submatrices
            supmat = self.tile_submatrices(CENMULT, SUBMAT_DICT)
            return supmat

        return compute_supermatrix

    
    def tile_submatrices(self, CENMULT, SM):
        """Function to loop over the submatrix blocks and tile in the 
        submatrices into the supermatrix.
        """
        supmat = jnp.zeros((CENMULT.dim_super, CENMULT.dim_super), dtype='float32')

        for i in range(CENMULT.dim_blocks):
            for ii in range(i, CENMULT.dim_blocks):
                startx, starty = SM.startx[i,ii], SM.starty[i,ii]
                endx, endy = SM.endx[i,ii], SM.endy[i,ii]

                # creating the submatrix
                submat = jnp.ones((endx-startx, endy-starty), dtype='float32')

                supmat = jax.ops.index_update(supmat,
                                              jax.ops.index[startx:endx, starty:endy],
                                              submat)
                # to avoid repeated filling of the central blocks
                supmat = jax.lax.cond(abs(i-ii)>0,
                                      lambda __: jax.ops.index_update(supmat,
                                                           jax.ops.index[starty:endy, startx:endx],
                                                           jnp.transpose(jnp.conjugate(submat))),
                                      lambda __: supmat,
                                      operand=None)

        return supmat
        
#####################################################################################

# initializing the necessary variables in gvar
fwindow = 150   # in muHz
n0, ell0 = 0, 200  # central multiplet
smax = 5    # max degree of perturbation (considering odd only)
cenmult_idx = 3672   # hardcoding central multiplet index for (0, 200)

# hardcong unperturbed freq of central multiplet
# needs to be scaled up by GVAR.OM * 1e6 to be in muHz
unit_omega = 20.963670602632025   # GVAR.OM * 1e6  
omegaref = 67.99455100807411 

# loading the neighbours. Here, we have hardcoded the following multiplets
# (0,198), (0,200), (0,202). 
nl_neighbours = np.array([[0,198], [0,200], [0,202]], dtype='int32')
nl_neighbours_idx = np.array([3650, 3672, 3693], dtype='int32')
omega_neighbours = np.array([67.65916460412984, 67.99455100807411, 68.32826640883721])

# computing the dimensions of the supermatrix apriori since this should be a static 
# variable and not a traced value
dim_super = 2 * np.sum(nl_neighbours[:,1] + 1)
dim_blocks = np.size(omega_neighbours)

# supermatix can be tiled with submatrices corresponding to                                                                                                      
# (l, n) - (l', n') coupling. The dimensions of the submatrix                                                                                                    
# is (2l+1, 2l'+1)                                                                                                                                               
dimX_submat = 2*nl_neighbours[:, 1].reshape(1, dim_blocks) \
              * np.ones((dim_blocks, 1), dtype='int32') + 1
dimY_submat = 2*nl_neighbours[:, 1].reshape(dim_blocks, 1) \
              * np.ones((1, dim_blocks), dtype='int32') + 1

# creating the startx, startx, endx, endy for subnatrices
submat_tile_ind = np.zeros((dim_blocks, dim_blocks, 4), dtype='int32')
for ix in range(dim_blocks):
    for iy in range(dim_blocks):
        submat_tile_ind[ix,iy,0] = int(dimX_submat[0, :ix].sum()) 
        submat_tile_ind[ix,iy,1] = int(dimY_submat[:iy, 0].sum())
        submat_tile_ind[ix,iy,2] = int(dimX_submat[0, :int(ix+1)].sum()) 
        submat_tile_ind[ix,iy,3] = int(dimY_submat[:int(iy+1), 0].sum())

# converting numpy to jax.numpy type arrays
nl_neighbours = jnp.array(nl_neighbours)
nl_neighbours_idx = jnp.array(nl_neighbours_idx)
omega_neighbours = jnp.array(omega_neighbours)

dimX_submat = jnp.array(dimX_submat)
dimY_submat = jnp.array(dimY_submat)

#######################################################################################
# CREATING THE NAMED TUPLES

# this dictionary does not change with changing central multiplets
GVAR_ = namedtuple('GVAR', 'fwindow smax unit_omega')

# this dictionary changes with central multiplet. NEEDS TO BE A STATIC ARGUMENT
CENMULT_ = namedtuple('CENMULT', 'dummy n0 ell0 omegaref cenmult_idx dim_super dim_blocks')

# dictionary for neighbours. WILL CONTAIN ARRAYS. CAN'T BE STATIC ARGUMENT
NEIGHBOUR_DICT_ = namedtuple('NEIGHBOUR_DICT', 'nl_neighbours nl_neighbours_idx, omega_neighbours,\
                             dimX_submat, dimY_submat')

SUBMAT_DICT_ = namedtuple('SUBMAT_DICT', 'startx starty endx endy')

# INITIALIZING THE NAMEDTUPLES
GVAR = GVAR_(fwindow, smax, unit_omega)
CENMULT = CENMULT_(-1, n0, ell0, omegaref, cenmult_idx, dim_super, dim_blocks)
NEIGHBOUR_DICT = NEIGHBOUR_DICT_(nl_neighbours, nl_neighbours_idx, omega_neighbours, dimX_submat, dimY_submat)

SUBMAT_DICT = SUBMAT_DICT_(submat_tile_ind[:,:,0], submat_tile_ind[:,:,1], submat_tile_ind[:,:,2], submat_tile_ind[:,:,3])

# creating instance of the class qdptMode
qdpt_mode = qdptMode(GVAR)

# jitting function to compute supermatrix
_compute_supermatrix = gnool_jit(qdpt_mode.build_supermatrix_function(), static_array_argnums=(0,2))

t1 = time.time()
for dummy in range(200):
    CENMULT = CENMULT_(dummy, n0, ell0, omegaref, cenmult_idx, dim_super, dim_blocks)
    # print('Executing dummy: ', dummy)
    __ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT, SUBMAT_DICT).block_until_ready()
t2 = time.time()
print(f'Compiling 200 supermatrices in: {t2-t1} seconds')

t3 = time.time()
for dummy in range(200):
    CENMULT = CENMULT_(dummy, n0, ell0, omegaref, cenmult_idx, dim_super, dim_blocks)
    # print('Executing dummy: ', dummy)
    __ = _compute_supermatrix(CENMULT, NEIGHBOUR_DICT, SUBMAT_DICT).block_until_ready()
t4 = time.time()
print(f'Executing 200 supermatrices in: {t4-t3} seconds.')
