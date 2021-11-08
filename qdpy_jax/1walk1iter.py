import numpy as np
import time
import sys

import jax
import jax.numpy as jnp
import jax.tree_util as tu

# new package in jax.numpy
from qdpy_jax import build_cenmult_and_nbs as build_CENMULT_AND_NBS 
from qdpy_jax import build_supermatrix as build_supmat
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import load_multiplets
from qdpy_jax import jax_functions as jf
from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import prune_multiplets

jax.config.update('jax_platform_name', 'cpu')

# enabling 64 bits

from jax.config import config
config.update('jax_enable_x64', True)

GVARS = gvar_jax.GlobalVars()
GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()

# jitting various functions
get_namedtuple_for_cenmult_and_neighbours = build_CENMULT_AND_NBS.get_namedtuple_for_cenmult_and_neighbours

build_SUBMAT_INDICES = build_supmat.build_SUBMAT_INDICES

# initialzing the class instance for supermatrix computation
build_supmat_funcs = build_supmat.build_supermatrix_functions()    
build_supermatrix = build_supmat_funcs.get_func2build_supermatrix()

# extracting the pruned parameters for multiplets of interest
nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
                prune_multiplets.get_pruned_attributes(GVARS, GVARS_ST)

lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                     nl_idx_pruned,
                                     omega_pruned)

GVARS_PRUNED_TR = jf.create_namedtuple('GVARS_TR',
                                       ['r',
                                        'r_spline',
                                        'rth',
                                        'wsr',
                                        'U_arr',
                                        'V_arr',
                                        'wig_list',
                                        'ctrl_arr_up',
                                        'ctrl_arr_lo',
                                        'knot_arr'],
                                       (GVARS_TR.r,
                                        GVARS_TR.r_spline,
                                        GVARS_TR.rth,
                                        GVARS_TR.wsr,
                                        lm.U_arr,
                                        lm.V_arr,
                                        wig_list,
                                        GVARS_TR.ctrl_arr_up,
                                        GVARS_TR.ctrl_arr_lo,
                                        GVARS_TR.knot_arr))

GVARS_PRUNED_ST = jf.create_namedtuple('GVARS_ST',
                                       ['s_arr',
                                        'nl_all',
                                        'nl_idx_pruned',
                                        'omega_list',
                                        'fwindow',
                                        'OM',
                                        'wig_idx',
                                        'rth_ind',
                                        'spl_deg'],
                                       (GVARS_ST.s_arr,
                                        lm.nl_pruned,
                                        lm.nl_idx_pruned,
                                        lm.omega_pruned,
                                        GVARS_ST.fwindow,
                                        GVARS_ST.OM,
                                        wig_idx,
                                        GVARS_ST.rth_ind,
                                        GVARS_ST.spl_deg))

nmults = len(GVARS.n0_arr)

def model():
    ev_sum = 0.0

    c1_list = []
    c3_list = []
    c5_list = []
    for i in range(GVARS_TR.ctrl_arr_up.shape[1]):
        c1_list.append(GVARS_TR.ctrl_arr_up[0, i])
        c3_list.append(GVARS_TR.ctrl_arr_up[1, i])
        c5_list.append(GVARS_TR.ctrl_arr_up[2, i])

    ctrl_arr = [jnp.array(c1_list),
                jnp.array(c3_list),
                jnp.array(c5_list)]

    for i in range(nmults):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)
        CENMULT_AND_NBS = jf.tree_map_CNM_AND_NBS(CENMULT_AND_NBS)

        SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
        SUBMAT_DICT = jf.tree_map_SUBMAT_DICT(SUBMAT_DICT)

        supmatrix = build_supermatrix(CENMULT_AND_NBS,
                                      SUBMAT_DICT,
                                      GVARS_PRUNED_ST,
                                      GVARS_PRUNED_TR,
                                      ctrl_arr)
        eigvals, eigvecs = jnp.linalg.eigh(supmatrix)
        eigvals = build_supmat.eigval_sort_slice(eigvals, eigvecs)
        ev_sum += jnp.sum(eigvals[:2*GVARS.ell0_arr[i]+1])
        print(f'Calculated supermatrix for multiplet = ({n0}, {ell0})')
    return ev_sum

if __name__ == "__main__":
    # jitting model()
    model_ = jax.jit(model)

    # COMPILING JAX
    # looping over the ells
    t1c = time.time()
    __ = model_().block_until_ready()
    t2c = time.time()
    print(f'Time taken in seconds for compilation of {nmults} multiplets' +
        f' =  {t2c-t1c:.2f} seconds')

    print("--------------------------------------------------")

    # EXECUTING JAX
    Niter = 1
    t1e = time.time()
    for i in range(Niter):
        __ = model_().block_until_ready()
    t2e = time.time()

    factor4niter = 1500 * 200./3600.
    t_projected_jit = (t2e-t1e) / nmults /Niter * factor4niter
    t_projected_eigval = 2. * factor4niter
    print(f'Time taken in seconds by jax-jitted execution' +
        f' of entire simulation (1500 iterations) = {t_projected_jit:.2f} hours')

    n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
    CENMULT_AND_NBS = get_namedtuple_for_cenmult_and_neighbours(n0, ell0, GVARS_ST)
    CENMULT_AND_NBS = jf.tree_map_CNM_AND_NBS(CENMULT_AND_NBS)

    SUBMAT_DICT = build_SUBMAT_INDICES(CENMULT_AND_NBS)
    SUBMAT_DICT = jf.tree_map_SUBMAT_DICT(SUBMAT_DICT)

    supmatrix = build_supermatrix(CENMULT_AND_NBS,
                                  SUBMAT_DICT,
                                  GVARS_PRUNED_ST,
                                  GVARS_PRUNED_TR,
                                  GVARS_TR.ctrl_arr_dpt)
    np.save('supmat-jax2-200.npy', np.array(supmatrix))
