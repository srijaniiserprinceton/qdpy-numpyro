import jax
import jax.tree_util as tu
from collections import namedtuple
import numpy as np
import time
import sys

# new package in jax.numpy
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat
from qdpy_jax import globalvars as gvar_jax

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
del GVARS


# namedtuple containing the central multiplets and its frequencies
CENMULT_ = namedtuple('CENMULT', ['n0_arr',
                                  'ell0_arr',
                                  'omega0_arr'])

CENMULT = CENMULT_(GVARS_ST.nl_pruned[:, 0],
                   GVARS_ST.nl_pruned[:, 1],
                   GVARS_ST.omega_pruned)

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours_ = \
    gjit.gnool_jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                   static_array_argnums = (0, 1, 2))

build_SUBMAT_INDICES_ = gjit.gnool_jit(build_supmat.build_SUBMAT_INDICES,
                                       static_array_argnums=(0,))

# initialzing the class instance for subpermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()    
build_supermatrix_ = gjit.gnool_jit(build_supmat_funcs.get_func2build_supermatrix(),
                                    static_array_argnums=(0,1,2))

# COMPILING JAX
# looping over the ells
t1c = time.time()

for i in range(GVARS_TR.nmults):
    n0, ell0, omega0 = CENMULT.n0_arr[i], CENMULT.ell0_arr[i], CENMULT.omega0_arr[i]

    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0,
                                                                 GVARS_ST, GVARS_TR)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT,
                                   GVARS_ST, GVARS_TR).block_until_ready()
    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
t2c = time.time()
print(f'Time taken in seconds for compilation of {GVARS_TR.nmults} multiplets' +
      f' =  {t2c-t1c:.2f} seconds')

# EXECUTING JAX
t1e = time.time()
print("--------------------------------------------------")

for i in range(GVARS_TR.nmults):
    n0, ell0, omega0 = CENMULT.n0_arr[i], CENMULT.ell0_arr[i], CENMULT.omega0_arr[i]

    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0,
                                                                 GVARS_ST, GVARS_TR)
    CENMULT_AND_NBS = tu.tree_map(lambda x: np.array(x), CENMULT_AND_NBS)
    SUBMAT_DICT = build_SUBMAT_INDICES_(CENMULT_AND_NBS)
    SUBMAT_DICT = tu.tree_map(lambda x: np.array(x), SUBMAT_DICT)

    supmatrix = build_supermatrix_(CENMULT_AND_NBS, SUBMAT_DICT,
                                   GVARS_ST, GVARS_TR).block_until_ready()
    print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
t2e = time.time()

t_projected_jit = (t2e-t1e) / GVARS_TR.nmults * 1500 * 200./3600.
print(f'Time taken in seconds by jax-jitted execution' +
      f' of entire simulation (1500 iterations) = {t_projected_jit:.2f} hours')
