from collections import namedtuple
from jax.experimental import host_callback as hcall
import jax.tree_util as tu

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

def jax_print(*args):
    str = ""
    for arg in args:
        str += f"{hcall.id_print(arg)}" + " "
    return str

def tree_map_CNM_AND_NBS(CNM_AND_NBS):
    nl_nbs, nl_nbs_idx, num_nbs, dim_super = tu.tree_map(lambda x: int(x),
                                                         (CNM_AND_NBS.nl_nbs,
                                                          CNM_AND_NBS.nl_nbs_idx,
                                                          CNM_AND_NBS.num_nbs,
                                                          CNM_AND_NBS.dim_super))
    omega_nbs = tu.tree_map(lambda x: float(x), CNM_AND_NBS.omega_nbs)
    

    CENMULT_AND_NBS = create_namedtuple('CENMULT_AND_NBS',
                                        ['nl_nbs',
                                         'nl_nbs_idx',
                                         'omega_nbs',
                                         'num_nbs',
                                         'dim_super'],
                                        (nl_nbs,
                                         nl_nbs_idx,
                                         omega_nbs,
                                         num_nbs,
                                         dim_super))

    
    return CENMULT_AND_NBS

def tree_map_SUBMAT_DICT(SUBMAT_DICT):
    # converting to nested tuples first
    SUBMAT_DICT = tu.tree_map(lambda x: tuple(map(tuple, x)), SUBMAT_DICT)

    # getting rid of the DeviceArray
    SUBMAT_DICT = tu.tree_map(lambda x: int(x), SUBMAT_DICT)

    return SUBMAT_DICT
