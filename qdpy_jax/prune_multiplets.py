from jax import jit
from tqdm import tqdm
import numpy as np

from qdpy import wigner_map as wigmap
from qdpy_jax import build_cenmult_and_nbs as build_cnm

# jitting various functions
getnt4cenmult_ = jit(build_cnm.getnt4cenmult, static_argnums = (0, 1, 2))

# slices out the unique nl, nl_idx and omega from
# from the arguments nl, omega which may contain repetitions
def get_pruned_multiplets(nl, omega, nl_all):
    n1 = nl[:, 0]
    l1 = nl[:, 1]

    omega_pruned = [omega[0]]
    nl_idx_pruned = [nl_all.tolist().index([nl[0, 0], nl[0, 1]])]
    nl_pruned = nl[0, :].reshape(1, 2)

    for i in range(1, len(n1)):
        try:
            nl_pruned.tolist().index([n1[i], l1[i]])
        except ValueError:
            nl_pruned = np.concatenate((nl_pruned,
                                        nl[i, :].reshape(1, 2)), 0)
            omega_pruned.append(omega[i])
            nl_idx_pruned.append(nl_all.tolist().index([nl[i, 0], nl[i, 1]]))
    return nl_pruned, nl_idx_pruned, omega_pruned


def get_pruned_attributes(GVARS):
    wig_list = []
    wig_idx = []
    
    for i in tqdm(range(len(GVARS.n0_arr)), desc=f"Precomputing wigners..."):
        n0, ell0 = GVARS.n0_arr[i], GVARS.ell0_arr[i]
        
        # building the namedtuple for the central multiplet and its neighbours
        CENMULT_AND_NBS = getnt4cenmult_(n0, ell0, GVARS)
        
        if i == 0:
            nl_pruned = CENMULT_AND_NBS.nl_nbs
            omega_pruned = CENMULT_AND_NBS.omega_nbs
        else:
            nl_pruned = np.concatenate((nl_pruned, CENMULT_AND_NBS.nl_nbs), 0)
            omega_pruned = np.append(omega_pruned, CENMULT_AND_NBS.omega_nbs)
            
        wig_list, wig_idx = wigmap.get_wigners_qdpy(CENMULT_AND_NBS.nl_nbs,
                                                    GVARS.s_arr, wig_list, wig_idx)
        

    nl_arr = np.asarray(GVARS.nl_all)
    nl_pruned = np.asarray(nl_pruned)

    # extracting the unique multiplets in the nl_pruned
    nl_pruned, nl_idx_pruned, omega_pruned = get_pruned_multiplets(nl_pruned,
                                                                   omega_pruned,
                                                                   nl_arr)
    
    nl_pruned = np.array(nl_pruned).astype('int')
    nl_idx_pruned = np.array(nl_idx_pruned).astype('int')
    omega_pruned = np.array(omega_pruned)
    wig_list = np.array(wig_list)
    wig_idx = np.array(wig_idx)

    # sorting the wigner indices for binary search
    sortind_wig_idx = np.argsort(wig_idx)
    wig_idx = wig_idx[sortind_wig_idx]
    wig_list = wig_list[sortind_wig_idx]

    # converting to tuples and nested tuples
    # done only for the ones which will be static in other functions
    nl_pruned = tuple(map(tuple, nl_pruned))
    nl_idx_pruned = tuple(nl_idx_pruned)
    omega_pruned = tuple(omega_pruned)
    wig_idx = tuple(wig_idx)

    return nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx
