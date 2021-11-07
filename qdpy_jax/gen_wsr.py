import bsplines as bsp

def gen_wsr(r, c, t, k):
    wsr_1 = bsp.bspline1d(r, c[0], t, k) 
    wsr_2 = bsp.bspline1d(r, c[1], t, k)
    wsr_3 = bsp.bspline1d(r, c[2], t, k)
    
    return jnp.array([wsr_1, wsr_3, wsr_5])

def get_matching_function(self):
    return (np.tanh((self.r - self.rth)/0.05) + 1)/2.0

def create_nearsurface_profile(self, idx, which_ex='upex'):
    w_dpt = self.wsr_dpt[idx, :]
    w_new = np.zeros_like(w_dpt)

    matching_function = self.get_matching_function()

    if (which_ex == 'upex'): scale_factor = self.gvar.fac_up[idx]
    else: scale_factor = self.gvar.fac_lo[idx]

    # near surface enhanced or suppressed profile
    # & adding the complementary part below the rth
    w_new = matching_function * scale_factor * w_dpt
    w_new += (1 - matching_function) * w_dpt

    return w_new
