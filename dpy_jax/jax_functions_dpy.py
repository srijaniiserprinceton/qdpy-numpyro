from collections import namedtuple
# from jax.experimental import host_callback as hcall
import jax.tree_util as tu
import time
from tqdm import tqdm
from jax import jit
from jax.lax import cond as cond
import jax.numpy as jnp
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


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

def model_renorm(model_params, model_params_ref, sigma):
    model_renorm_arr = model_params / model_params_ref - 1.
    # model_renorm_arr = model_params - model_params_ref
    # model_renorm_arr = model_params
    model_renorm_arr = model_renorm_arr / sigma
    return model_renorm_arr

def model_denorm(model_params, model_params_ref, sigma):
    model_denorm_arr = (model_params * sigma + 1.) * model_params_ref
    # model_denorm_arr = model_params * sigma + model_params_ref
    # model_denorm_arr = model_params * sigma
    return model_denorm_arr

def model_renorm_log(model_params, model_params_ref, sigma):
    model_renorm_arr = jnp.log(model_params / model_params_ref) / sigma
    return model_renorm_arr

def model_denorm_log(model_params, model_params_ref, sigma):
    model_denorm_arr = jnp.exp(model_params * sigma) * model_params_ref
    return model_denorm_arr

def c4fit_2_c4plot(GVARS, c_arr, sind_arr, cind_arr):
    c_arr_plot_clipped = 1.0 * GVARS.ctrl_arr_dpt_clipped
    c_arr_plot_shaped = jnp.reshape(c_arr, (len(sind_arr), -1), 'F')
    
    for sind_idx, sind in enumerate(sind_arr):
        c_arr_plot_clipped[sind, cind_arr] = c_arr_plot_shaped[sind_idx]

    c_arr_plot_full = 1.0 * GVARS.ctrl_arr_dpt_full

    # tiling the fitted values in the larger array of ctrl points
    c_arr_plot_full[:, GVARS.knot_ind_th:] = c_arr_plot_clipped
    
    return c_arr_plot_full

def D(f, r):
    num_f = f.shape[0]
    d2r_df2 = np.zeros_like(f)
    for i in range(num_f):
        dr_df = np.gradient(f[i], r, edge_order=2)
        d2r_df2[i] = np.gradient(dr_df, r, edge_order=2)
        # d2r_df2[i] = dr_df
    
    return d2r_df2
    
def get_model_covariance(G_g_inv, C_d):
    return G_g_inv @ jnp.linalg.inv(C_d) @ G_g_inv.T
