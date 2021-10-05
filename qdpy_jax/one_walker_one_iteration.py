import jax
from qdpt_jax import gnool_jit as gjit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat

# the radial branch [choosing f-branch for testing purposes]
n0 = 0

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours_ = jax.jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))

# initialzing the class instance for subpermatrix computation
build_supmat_funcs = build_supmat.build_supmatrix_functions()    
build_supermatrix_ = gjit(build_supmat_funcs.build_supermatrix, static_array_argnums=())

# looping over the ells
for ell0 in range(195, 290):
    # building the namedtuple for the central multiplet and its neighbours
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)

    supmatrix = build_supmat_funcs.build_supermatrix()
    
