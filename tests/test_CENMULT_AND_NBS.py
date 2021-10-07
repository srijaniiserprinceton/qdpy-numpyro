import time 
from jax import jit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS_jax
from qdpy_numpy import build_cenmult_and_nbs as build_CENMULT_AND_NBS_np                                                                                                                

def test_build_CENMULT_AND_NBS():
    
    # obtaining the namedtuple for the central mode and its neighbours                                                                                                         
    get_namedtuple_for_cenmult_and_neighbours_ = jit(build_CENMULT_AND_NBS_jax.get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))

    # defining the radial order
    n0 = 0

    #-----------------------------------------------------------------------------------------------
    # RUNNING IN STANDARD NUMPY
    t1n = time.time()                                                                                                                                                               
    for ell0 in range(195, 290):                                                                                                                                               
        CENMULT_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)                                                                                  
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print("[NumPy]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2n = time.time()                        
    print(f'Computing in: {t2n-t1n} seconds')
    
    #------------------------------------------------------------------------------------------------
    # COMPILING JAX
    t1c = time.time()                                                                                                                                                               
    for ell0 in range(195, 290):                                                                                                                                               
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)                                                                                                       
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print("[JAX]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2c = time.time()                                                                                                                                                               
    print(f'Compiling in: {t2c-t1c} seconds')                                                                                                                                        
        
    #------------------------------------------------------------------------------------------------
    # EXECUTING JAX
    t1e = time.time()                                                                                                                                                            
    for ell0 in range(195, 290):                                                                                                                                                     
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)                                                                                                        
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple    
    print("[JAX-JIT]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2e = time.time()
    print(f'Executing in: {t2e-t1e} seconds.')                                                                                                                                         

    #-----------------------------------------------------------------------------------------------
    print(f'Speedup of JAX-JIT vs. JAX: {(t2c-t1c)/(t2e-t1e)}')
    print(f'Speedup of JAX-JIT vs. NumPy: {(t2n-t1n)/(t2e-t1e)}')

    te = t2e - t1e
    tn = t2n - t1n

    if(te < tn):
        print('USE JAX-JIT!')
    else:
        print('USE NUMPY!')

if __name__ == '__main__':
    test_build_CENMULT_AND_NBS()
