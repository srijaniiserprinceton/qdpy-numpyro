from collections import namedtuple
from jax.experimental import host_callback as hcall

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

def jax_print(*args):
    str = ""
    for arg in args:
        str += f"{hcall.id_print(arg)}" + " "
    return str
