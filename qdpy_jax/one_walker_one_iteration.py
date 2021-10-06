import jax
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat
import numpy as np
from collections import namedtuple

# the radial branch [choosing f-branch for testing purposes]
# this array of multiplets needs to be built separately from hmi file
ell0_arr = np.arange(195, 290, dtype='int32')
n0_arr = np.zeros_like(ell0_arr, dtype='int32')

NMULTS = len(ell0_arr)

# finding the omega0 for all the multiplets
omega0_arr = np.zeros(len(ell0_arr))

for i in range(NMULTS):
    omega0_arr[i] = 1

# namedtuple containing the central multiplet and its frequencies
CENMULT_ = namedtuple('CENMULT', ['n0_arr',
                                  'ell0_arr',
                                  'omega0_arr'])

CENMULT = CENMULT_(n0_arr, ell0_arr, omega0_arr)

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours_ = jax.jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))

# initialzing the class instance for subpermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()    
build_supermatrix_ = gjit.gnool_jit(build_supmat_funcs.get_func2build_supermatrix(), static_array_argnums=(0,1))

# looping over the ells
for i in range(NMULTS):
    n0, ell0, omega0 = CENMULT.n0_arr[i], CENMULT.ell0_arr[i], CENMULT.omega0_arr[i]
    
    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)
    
    SUBMAT_DICT = build_supmat.build_SUBMAT_INDICES(CENMULT_AND_NBS)

    print(SUBMAT_DICT.startx[5,5])
    continue
    
    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT)
    
