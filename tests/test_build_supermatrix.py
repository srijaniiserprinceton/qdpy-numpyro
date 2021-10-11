import time 
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_supermatrix as build_supermatrix_jax
from qdpy_numpy import build_supermatrix as build_supermatrix_np
from qdpy_numpy import build_cenmult_and_nbs as build_CENMULT_AND_NBS_np

def test_build_supermatrix():
    
    build_supmat_funcs_np = build_supermatrix_np.build_supermatrix_functions().get_func2build_supermatrix()
    build_supmat_funcs_jax = build_supermatrix_jax.build_supermatrix_functions()
    build_supermatrix_ = gjit.gnool_jit(build_supmat_funcs_jax.get_func2build_supermatrix(),
                                        static_array_argnums=(0,1))

    # defining the radial order
    n0 = 0
    ell_min = 195
    ell_max = 210
    
    #-----------------------------------------------------------------------------------------------
    # RUNNING IN STANDARD NUMPY
    t1n = time.time()
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)
        SUBMAT_DICT = build_supermatrix_np.build_SUBMAT_INDICES(CNM_AND_NBS)                                                                                
        supmatrix = build_supmat_funcs_np(CNM_AND_NBS, SUBMAT_DICT)
        
    t2n = time.time()
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple
    print(f'[NumPy] Computing in: {(t2n-t1n)} seconds')
    
    #------------------------------------------------------------------------------------------------
    # COMPILING JAX
    t1c = time.time()
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)
        SUBMAT_DICT = build_supermatrix_np.build_SUBMAT_INDICES(CNM_AND_NBS)                          
        supmatrix = build_supermatrix_(CNM_AND_NBS, SUBMAT_DICT).block_until_ready()

    t2c = time.time()
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print(f'[JAX] Compiling in: {(t2c-t1c)} seconds')
    #------------------------------------------------------------------------------------------------
    # EXECUTING JAX
    t1e = time.time()
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)
        SUBMAT_DICT = build_supermatrix_np.build_SUBMAT_INDICES(CNM_AND_NBS)
        supmatrix = build_supermatrix_(CNM_AND_NBS, SUBMAT_DICT).block_until_ready()
    
    t2e = time.time()
        

    
    te = t2e-t1e
    tc = t2c-t1c
    tn = t2n-t1n

    
    print(f'[JAX-JIT] Executing in: {te} seconds.')
    #-----------------------------------------------------------------------------------------------
    print(f'Speedup of JAX-JIT vs. JAX: {(tc)/(te)}')
    print(f'Speedup of JAX-JIT vs. NumPy: {(tn)/(te)}')

    print('=========================================')

    if(te < tn):
        print('USE JAX-JIT!')
    else:
        print('USE NUMPY!')

if __name__ == '__main__':
    test_build_supermatrix()
