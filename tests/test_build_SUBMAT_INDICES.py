import time 
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import build_supermatrix as build_supermatrix_jax
from qdpy_numpy import build_supermatrix as build_supermatrix_np                                                                                                                
from qdpy_numpy import build_cenmult_and_nbs as build_CENMULT_AND_NBS_np

def test_build_SUBMAT_INDICES():
    
    # obtaining the namedtuple for the central mode and its neighbours                                                                                                         
    build_SUBMAT_INDICES_ = gjit.gnool_jit(build_supermatrix_jax.build_SUBMAT_INDICES,
                                                     static_array_argnums = (0,))

    # defining the radial order
    n0 = 0
    ell_min = 195
    ell_max = 210
    
    #-----------------------------------------------------------------------------------------------
    # RUNNING IN STANDARD NUMPY
    tn = 0.0              
    result = 0.0
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)

        tn1 = time.time()
        SUBMAT_DICT = build_supermatrix_np.build_SUBMAT_INDICES(CNM_AND_NBS)                                                                                 
        result += SUBMAT_DICT.startx[0,0]
        tn2 = time.time()
    
        tn += (tn2-tn1)
        
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple
    print("[NumPy]: ", result)
    print(f'Computing in: {tn} seconds')
    
    #------------------------------------------------------------------------------------------------
    # COMPILING JAX
    tc = 0.0
    result = 0.0
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)

        tc1 = time.time()
        SUBMAT_DICT = build_SUBMAT_INDICES_(CNM_AND_NBS)
        result += SUBMAT_DICT.startx[0,0]
        tc2 = time.time()
    
        tc += (tc2-tc1)
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print("[JAX]: ", result)
    print(f'Compiling in: {tc} seconds')

    #------------------------------------------------------------------------------------------------
    # EXECUTING JAX
    te = 0.0
    result = 0.0
    for ell0 in range(ell_min, ell_max+1):                                                                                                                                              
        # getting the argument for this function
        CNM_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)

        te1 = time.time()
        SUBMAT_DICT = build_SUBMAT_INDICES_(CNM_AND_NBS)                                                                                 
        result += SUBMAT_DICT.startx[0,0]
        te2 = time.time()

        te += (te2-te1)
        

    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple    
    print("[JAX-JIT]: ", result)
    print(f'Executing in: {te} seconds.')                                                                                                                                         

    #-----------------------------------------------------------------------------------------------
    print(f'Speedup of JAX-JIT vs. JAX: {(tc)/(te)}')
    print(f'Speedup of JAX-JIT vs. NumPy: {(tn)/(te)}')

    print('=========================================')

    if(te < tn):
        print('USE JAX-JIT!')
    else:
        print('USE NUMPY!')

if __name__ == '__main__':
    test_build_SUBMAT_INDICES()
