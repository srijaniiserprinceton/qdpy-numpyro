import bsplines as bsp

def gen_wsr(r, c, t, k):
    wsr_1 = bsp.bspline1d(r, c[0], t, k) 
    wsr_2 = bsp.bspline1d(r, c[1], t, k)
    wsr_3 = bsp.bspline1d(r, c[2], t, k)
    
    return jnp.array([wsr_1, wsr_3, wsr_5])
    
