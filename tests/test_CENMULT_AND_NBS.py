import time 
from jax import jit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS                                                                                                                                               

def test_build_CENMULT_AND_NBS():
    # defining the central mode (n0, ell0)                                                                                                                                                   
    n0 ,ell0 = 1, 150
    
    # obtaining the namedtuple for the central mode and its neighbours                                                                                                                       
    get_namedtuple_for_cenmult_and_neighbours_ = jit(build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours,
                                                     static_argnums = (0,1))
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0)
    
    
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

if __name__ == '__main__':
    test_build_CENMULT_AND_NBS()
