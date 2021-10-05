import jax
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
# the radial branch [choosing f-branch for testing purposes]
n0 = 0

# jitting the function
get_namedtuple_for_cenmult_and_neighbours_ = jax.jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))

# looping over the ells
for ell0 in range(195, 290):
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)
    
