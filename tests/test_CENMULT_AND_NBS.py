import time 
from jax import jit
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS_jax
from qdpy_numpy import build_cenmult_and_nbs as build_CENMULT_AND_NBS_np                                                                                                                
from qdpy_jax import globalvars
from qdpy_jax import gnool_jit as gjit

#------((( creating the namedtuples of global variables --------                                                                                                                        
GVARS = globalvars.GlobalVars()
# extracting only the required global variables                                                                                                                             
# for the multiplets of interest                                                                                                                                      
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
# deleting all large arrays which won't be used anymore                                                                                                                
del GVARS


def test_build_CENMULT_AND_NBS():
    
    # obtaining the namedtuple for the central mode and its neighbours                                                                                                         
    get_namedtuple_for_cenmult_and_neighbours_ = gjit.gnool_jit(build_CENMULT_AND_NBS_jax.get_namedtuple_for_cenmult_and_neighbours,
                                                                static_array_argnums = (0,1,2))

    # defining the radial order
    n0 = 0

    #-----------------------------------------------------------------------------------------------
    # RUNNING IN STANDARD NUMPY
    t1n = time.time()                                                                                                                                                               
    for i in range(GVARS_TR.nmults):                                                                                                                                               
        n0, ell0 = GVARS_ST.nl_pruned[i, 0], GVARS_ST.nl_pruned[i, 1]
        CENMULT_AND_NBS = build_CENMULT_AND_NBS_np.get_namedtuple_for_cenmult_and_neighbours(n0, ell0)                                                                                  
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print("[NumPy]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2n = time.time()                        
    print(f'Computing in: {t2n-t1n} seconds')
    
    #------------------------------------------------------------------------------------------------
    # COMPILING JAX
    t1c = time.time()                                                                                                                                                               
    for i in range(GVARS_TR.nmults):
        n0, ell0 = GVARS_ST.nl_pruned[i, 0], GVARS_ST.nl_pruned[i, 1]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST, GVARS_TR)                                                                                   
        
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple 
    print("[JAX]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2c = time.time()                                                                                                                                                               
    print(f'Compiling in: {t2c-t1c} seconds')                                                                                                                                        
        
    #------------------------------------------------------------------------------------------------
    # EXECUTING JAX
    t1e = time.time()                                                                                                                                                            
    for i in range(GVARS_TR.nmults):
        n0, ell0 = GVARS_ST.nl_pruned[i, 0], GVARS_ST.nl_pruned[i, 1]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours_(n0, ell0, GVARS_ST, GVARS_TR)                                                                                     
    
    # Arbitrary printing of a value since block_until_ready doesn't work for namedtuple    
    print("[JAX-JIT]: ", CENMULT_AND_NBS.nl_nbs[0])
    t2e = time.time()
    print(f'Executing in: {t2e-t1e} seconds.')                                                                                                                                         

    #-----------------------------------------------------------------------------------------------
    print(f'Speedup of JAX-JIT vs. JAX: {(t2c-t1c)/(t2e-t1e)}')
    print(f'Speedup of JAX-JIT vs. NumPy: {(t2n-t1n)/(t2e-t1e)}')

    te = t2e - t1e
    tn = t2n - t1n

    # the preferred method                                                                                                                                                              
    t_preferred = 0.0
    if(te < tn):
        t_preferred = te
        print('USE JAX-JIT!')
    else:
        t_preferred = tn
        print('USE NUMPY!')


    # the total contribution to compute time for 1500 iterations of MCMC                                                                                                                
    print('Total time taken for 1500 iterations: ', (t_preferred * 1500./(60.)), ' minutes.')


if __name__ == '__main__':
    test_build_CENMULT_AND_NBS()
