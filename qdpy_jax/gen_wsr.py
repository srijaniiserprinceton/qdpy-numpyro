import bsplines as bsp_adams
import jax
import jax.numpy as jnp

bspline = bsp_adams.bspline1d

jidx = jax.ops.index
jidx_update = jax.ops.index_update

def get_wsr_from_spline(r_spline, wsr_dpt, ctrl_arr,
                        knot_arr, rth_ind, spl_deg=3):
    wsr_new = jnp.zeros_like(wsr_dpt)
    wsr_1 = bspline(r_spline, ctrl_arr[0],
                    knot_arr, spl_deg)
    wsr_3 = bspline(r_spline, ctrl_arr[1],
                    knot_arr, spl_deg)
    wsr_5 = bspline(r_spline, ctrl_arr[2],
                    knot_arr, spl_deg)
    wsr_new = jidx_update(wsr_new, jidx[:, :rth_ind],
                          wsr_dpt[:, :rth_ind])
    wsr_new = jidx_update(wsr_new, jidx[0, rth_ind:], wsr_1)
    wsr_new = jidx_update(wsr_new, jidx[1, rth_ind:], wsr_3)
    wsr_new = jidx_update(wsr_new, jidx[2, rth_ind:], wsr_5)
    return wsr_new
