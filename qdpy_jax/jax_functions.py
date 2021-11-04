from collections import namedtuple
from jax.experimental import host_callback as hcall
import jax.tree_util as tu
from functools import partial
from jax import jit

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

def jax_print(*args):
    str = ""
    for arg in args:
        str += f"{hcall.id_print(arg)}" + " "
    return str

# @partial(jit, static_argnums=(0,))
def tree_map_CNM_AND_NBS(CNM_AND_NBS):
    # converting to tuples and nestes tuples for easy of passing
    nl_nbs = tuple(map(tuple, CNM_AND_NBS.nl_nbs))
    nl_nbs_idx = tuple(CNM_AND_NBS.nl_nbs_idx)
    omega_nbs = tuple(CNM_AND_NBS.omega_nbs)
    num_nbs = int(CNM_AND_NBS.num_nbs)
    dim_super = int(CNM_AND_NBS.dim_super)

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

    return SUBMAT_DICT
