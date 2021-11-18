from collections import namedtuple
# from jax.experimental import host_callback as hcall
import jax.tree_util as tu
import time
from tqdm import tqdm
from jax import jit
from jax.lax import cond as cond
import jax.numpy as jnp

def time_run(f, *args, unit="seconds", prefix="execution", Niter=1,
             block_until_ready=False):
    t1 = time.time()
    for i in tqdm(range(Niter)):
        if block_until_ready:
            __ = f(*args).block_until_ready()
        else:
            __ = f(*args)
    t2 = time.time()
    tdiff = t2 - t1
    if unit == "minutes":
        tdiff /= 60.
    elif unit == "hours":
        tdiff /= 3600.
    elif unit == "days":
        tdiff /= (3600.*24.)
    print(f"[{prefix}] Time taken = {tdiff/Niter:.3f} {unit}")

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

"""
def jax_print(*args):
    str = ""
    for arg in args:
        str += f"{hcall.id_print(arg)}" + " "
    return str
"""

def tree_map_CNM_AND_NBS(CNM_AND_NBS):
    # converting to tuples and nestes tuples for easy of passing
    nl_nbs = tuple(map(tuple, CNM_AND_NBS.nl_nbs))
    nl_nbs_idx = tuple(CNM_AND_NBS.nl_nbs_idx)
    omega_nbs = tuple(CNM_AND_NBS.omega_nbs)

    CENMULT_AND_NBS = create_namedtuple('CENMULT_AND_NBS',
                                        ['nl_nbs',
                                         'nl_nbs_idx',
                                         'omega_nbs'],
                                        (nl_nbs,
                                         nl_nbs_idx,
                                         omega_nbs))

    return CENMULT_AND_NBS

def tree_map_SUBMAT_DICT(SUBMAT_DICT):
    # converting to nested tuples first
    SUBMAT_DICT = tu.tree_map(lambda x: tuple(map(tuple, x)), SUBMAT_DICT)

    return SUBMAT_DICT

def jax_Omega(ell, N):
    """Computes Omega_N^\ell"""
    return cond(abs(N) > ell,
                lambda __: 0.0,
                lambda __: jnp.sqrt(0.5 * (ell+N) * (ell-N+1)),
                operand=None)
    
def jax_minus1pow_vec(num):
    """Computes (-1)^n"""
    modval = num % 2
    return (-1)**modval

def jax_gamma(ell):
    """Computes gamma_ell"""
    return jnp.sqrt((2*ell + 1)/4/jnp.pi)
