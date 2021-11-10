import numpy as np
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

get_namedtuple_for_cenmult_and_neighbours =\
                    build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours

def get_trace_arr():
    dim_hyper = 0
    num_nbs_total = 0

    nmults = len(GVARS.n0_arr)

    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)
        
        # dim_super of local supermatrix
        dim_super = np.sum(2*CENMULT_AND_NBS.nl_nbs[:, 1] + 1)

        if(dim_super > dim_hyper): dim_hyper = dim_super

        num_nbs_total += len(CENMULT_AND_NBS.omega_nbs)

    
    trace_arr = np.zeros((num_nbs_total, dim_hyper), dtype='bool')
    
    nbs_total_count = 0
    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)
        num_nbs = len(CENMULT_AND_NBS.omega_nbs)

        # SUBMAT_DICT stuff
        dimX_submat = 2 * CENMULT_AND_NBS.nl_nbs[:, 1] + 1
        
        startx_arr = np.cumsum(dimX_submat)[:-1]
        endx_arr = np.cumsum(dimX_submat)

        startx_arr = np.append([0], startx_arr)
        
        print(ell0, startx_arr, endx_arr)

        for tr_i in range(num_nbs):
            startx, endx = startx_arr[tr_i], endx_arr[tr_i]
            trace_arr[nbs_total_count, startx:endx] = 1

            nbs_total_count += 1

    return trace_arr, dim_hyper
        

trace_arr, dim_hyper = get_trace_arr()
np.save('trace_arr.npy', trace_arr)
