import jax
import jax.tree_util as tu
from collections import namedtuple
import numpy as np
import time

'''
# old package in numpy
import qdPy
from qdPy import globalvars as gvar
from qdPy import qdclasses as qdcls
from qdPy import w_Bsplines as w_Bsp
'''

# new package in jax.numpy
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat

# the radial branch [choosing f-branch for testing purposes]
# this array of multiplets needs to be built separately from hmi file
# ell0_arr = np.arange(195, 290, dtype='int32')
ell0_arr = np.arange(195, 200, dtype='int32')
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

build_SUBMAT_INDICES_ = gjit.gnool_jit(build_supmat.build_SUBMAT_INDICES, static_array_argnums=(0,))

# initialzing the class instance for subpermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()    
build_supermatrix_ = gjit.gnool_jit(build_supmat_funcs.get_func2build_supermatrix(), static_array_argnums=(0,1))

# COMPILING JAX
# looping over the ells
t1c = time.time()

for i in range(NMULTS):
    n0, ell0, omega0 = CENMULT.n0_arr[i], CENMULT.ell0_arr[i], CENMULT.omega0_arr[i]

    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    
    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)
    
    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT).block_until_ready()

    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')

t2c = time.time()

# EXECUTING JAX
t1e = time.time()

for i in range(NMULTS):
    n0, ell0, omega0 = CENMULT.n0_arr[i], CENMULT.ell0_arr[i], CENMULT.omega0_arr[i]

    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)

    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT).block_until_ready()

    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
    
t2e = time.time()


'''
class Args():
    def __init__(self,
                 n0=0,
                 l0=200,
                 lmin=195,
                 lmax=290,
                 maxiter=1,
                 precompute=False,
                 parallel=False):
        self.n0 = n0
        self.l0 = l0
        self.lmin = lmin
        self.lmax = lmax
        self.maxiter = maxiter
        self.precompute = precompute
        self.parallel = parallel

ARGS = Args(n0=n0_arr[0],
            l0=ell0_arr[0],
            lmin=ell0_arr.min(),
            lmax=ell0_arr.max())
GVAR = gvar.globalVars()
SPL_DICT = w_Bsp.wsr_Bspline(GVAR)

t1 = time.time()

for ell in ell0_arr:
    ARGS.l0 = ell
    analysis_modes = qdcls.qdptMode(GVAR, SPL_DICT)
    
    # print(analysis_modes.nl_neighbors,
          # analysis_modes.nl_neighbors_idx,
          # analysis_modes.omega_neighbors)
    
    supmat = analysis_modes.create_supermatrix()
    
    # print(supmat.dimX_submat,
          # supmat.dimY_submat)
    

t2 = time.time()

print(f'[jax-compile] Time taken = {(t2c-t1c):.3e} seconds')
print(f'[jax-execute] Time taken = {(t2e-t1e):.3e} seconds')
print(f'[  numpy    ] Time taken = {(t2-t1):.3e} seconds')
print(f'Speedup = {(t2-t1)/(t2e-t1e):.3f}x')

'''

print(f'[jax-compile] Time taken = {(t2c-t1c):.3e} seconds')
print(f'[jax-execute] Time taken = {(t2e-t1e):.3e} seconds')
