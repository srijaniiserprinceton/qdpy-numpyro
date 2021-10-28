import numpy as np
import time
import jax
import jax.numpy as jnp
from collections import namedtuple
import sys
import os
from functools import partial
from jax.lax import fori_loop as foril

from qdpy_jax import wigner_map2 as wigmap
from qdpy_jax import gnool_jit as gjit
from qdpy_jax import prune_multiplets
from qdpy_jax import load_multiplets
from qdpy_jax import globalvars as gvar_jax
from qdpy_jax import jax_functions as jf

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"


def jax_Omega(ell, N):
    """Computes Omega_N^\ell"""
    return jax.lax.cond(
        abs(N) > ell,
        lambda __: 0.0,
        lambda __: jnp.sqrt(0.5 * (ell+N) * (ell-N+1)),
        operand=None)
    
def jax_minus1pow(num):
    """Computes (-1)^n"""
    return jax.lax.cond(
        num % 2 == 1,
        lambda __: -1,
        lambda __: 1,
        operand=None)

def jax_minus1pow_vec(num):
    """Computes (-1)^n"""
    modval = num % 2
    return (-1)**modval


def jax_gamma(ell):
    """Computes gamma_ell"""
    return jnp.sqrt((2*ell + 1)/4/jnp.pi)

# _find_idx = gjit.gnool_jit(wigmap.find_idx, static_array_argnums=(3,))
_find_idx = jax.jit(wigmap.find_idx)

class compute_submatrix:
    def __init__(self, gvars):
        self.r = gvars.r
        self.s_arr = jnp.array(gvars.s_arr)
        self.wsr = gvars.wsr

    def jax_get_Cvec(self):
        def get_func_Cvec(qdpt_mode, eigfuncs, wigs):
            """Computing the non-zero components of the submatrix"""

            lenidx = len(wigs.wig_idx_full)

            ell1 = qdpt_mode.ell1
            ell2 = qdpt_mode.ell2
            ell = qdpt_mode.ellmin
            m = jnp.arange(-ell, ell+1)
            len_m = len(m)
            len_s = jnp.size(self.s_arr)
            wigvals = jnp.zeros((len_m, len_s))

            def modify_wig_ell(iem, func_params):
                wigvals_iem, idx1, idx2, fac = func_params
                idx = jnp.argmin(jnp.abs(wigs.wig_idx_full[:,0]-idx1[iem])
                                 + jnp.abs(wigs.wig_idx_full[:,1]-idx2[iem]))

                wigvals_iem = jax.ops.index_update(wigvals_iem,
                                               jax.ops.index[idx],
                                               fac[idx] * wigs.wig_list[idx])  

                return (wigvals_iem, idx1, idx2, fac)

            for i, s in enumerate(self.s_arr):
                idx1, idx2, fac = _find_idx(ell1, s, ell2, m)
                
                wigvals_iem = jnp.zeros((len_m))
                wigvals_iem, __, __, __ = foril(0, len_m, modify_wig_ell,
                                                (wigvals_iem, idx1, idx2, fac))

                wigvals = jax.ops.index_update(wigvals,
                                               jax.ops.index[:, i],
                                               wigvals_iem)
                
            Tsr = self.jax_compute_Tsr(qdpt_mode, eigfuncs)
            # -1 factor from definition of toroidal field
            '''wsr = np.loadtxt(f'{self.sup.gvar.datadir}/{WFNAME}')\
            [:, self.rmin_idx:self.rmax_idx] * (-1.0)'''
            # self.sup.spline_dict.get_wsr_from_Bspline()
            #wsr = self.sup.spline_dict.wsr
            # wsr[0, :] *= 0.0 # setting w1 = 0
            # wsr[1, :] *= 0.0 # setting w3 = 0
            # wsr[2, :] *= 0.0 # setting w5 = 0
            # wsr /= 2.0
            # integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]
            integrand = Tsr * self.wsr   # since U and V are scaled by sqrt(rho) * r

            #### TO BE REPLACED WITH SIMPSON #####
            integral = jnp.trapz(integrand, axis=1, x=self.r)

            prod_gammas = (jax_gamma(qdpt_mode.ell1) *
                           jax_gamma(qdpt_mode.ell2) *
                           jax_gamma(self.s_arr))
            omegaref = qdpt_mode.omegaref
            Cvec = (jax_minus1pow_vec(m) * 8*jnp.pi *
                    qdpt_mode.omegaref * (wigvals @ (prod_gammas * integral)))

            return Cvec

        return get_func_Cvec

    #def jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
    def jax_compute_Tsr(self, qdpt_mode, eigfuncs): 
        """Computing the kernels which are used for obtaining the
        submatrix elements.
        """
        Tsr = jnp.zeros((len(self.s_arr), len(self.r)))

        L1sq = qdpt_mode.ell1*(qdpt_mode.ell1+1)
        L2sq = qdpt_mode.ell2*(qdpt_mode.ell2+1)
        Om1 = jax_Omega(qdpt_mode.ell1, 0)
        Om2 = jax_Omega(qdpt_mode.ell2, 0)

        U1, U2, V1, V2 = eigfuncs.U1, eigfuncs.U2, eigfuncs.V1, eigfuncs.V2

        # creating internal function for the foril
        def func4Tsr_s_loop(i, iplist):
            Tsr, s_arr = iplist
            s = s_arr[i]
            ell1, ell2 = qdpt_mode.ell1, qdpt_mode.ell2
            r = self.r
            # s = self.s_arr[i]
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            # wigval = w3j(ell1, s, ell2, -1, 0, 1)
            # using some dummy number until we write the
            # function for mapping wigner3js
            wigval = 1.0
            Tsr_at_i = -(1 - jax_minus1pow(ell1 + ell2 + s)) * \
                       Om1 * Om2 * wigval * eigfac / r
            Tsr = jax.ops.index_update(Tsr, i, Tsr_at_i)

            return (Tsr, s_arr)

        Tsr, s_arr = foril(0, len(self.s_arr), func4Tsr_s_loop, (Tsr, self.s_arr))

        return Tsr


if __name__ == "__main__":
    '''
    # parameters to be included in the global dictionary later?
    s_arr = jnp.array([1,3,5], dtype='int32')

    rmin = 0.3
    rmax = 1.0

    r = np.loadtxt(f'{data_dir}/r.dat') # the radial grid

    # finding the indices for rmin and rmax
    rmin_ind = np.argmin(np.abs(r - rmin))
    rmax_ind = np.argmin(np.abs(r - rmax)) + 1

    # clipping radial grid
    r = r[rmin_ind:rmax_ind]

    # the rotation profile
    wsr = np.loadtxt(f'{data_dir}/w.dat')
    wsr = wsr[:,rmin_ind:rmax_ind]
    wsr = jnp.array(wsr)   # converting to device array once

    # using fixed modes (0,200)-(0,200) coupling for testing
    n1, n2 = 0, 0
    ell1, ell2 = 200, 200

    # finding omegaref
    omegaref = 1

    U = np.loadtxt(f'{data_dir}/U3672.dat')
    V = np.loadtxt(f'{data_dir}/V3672.dat')

    U = U[rmin_ind:rmax_ind]
    V = V[rmin_ind:rmax_ind]

    # converting numpy arrays to jax.numpy arrays
    r = jnp.array(r)
    U, V = jnp.array(U), jnp.array(V)

    U1, U2 = U, U
    V1, V2 = V, V

    # creating the named tuples
    GVAR = namedtuple('GVAR', ['r',
                               'wsr',
                               's_arr'])

    QDPT_MODE = namedtuple('QDPT_MODE', ['ell1',
                                         'ell2',
                                         'ellmin',
                                         'omegaref'])
    EIGFUNCS = namedtuple('EIGFUNCS', ['U1',
                                       'U2',
                                       'V1',
                                       'V2'])

    # initializing namedtuples. This could be done from a separate file later
    gvars = GVAR(r, wsr, s_arr)
    qdpt_mode = QDPT_MODE(ell1, ell2, min(ell1, ell2), omegaref)
    eigfuncs = EIGFUNCS(U1, U2, V1, V2)

    wigs = jf.create_namedtuple('WIGNERS',
                                ['wig_list',
                                 'wig_idx1',
                                 'wig_idx2',
                                 'wig_idx_full'],
                                (GVARS_TR.wig_list,
                                 GVARS_ST.wig_idx1,
                                 GVARS_ST.wig_idx2,
                                 GVARS_ST.wig_idx_full))
    '''

    GVARS = gvar_jax.GlobalVars()
    GVARS_PATHS, GVARS_TR, GVARS_ST = GVARS.get_all_GVAR()
    
    # extracting the pruned parameters for multiplets of interest                                                                                                              
    nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx_full = prune_multiplets.get_pruned_attributes(GVARS, GVARS_ST)


    # converting to list before sending into jax'd function
    # wig_idx_full = wig_idx_full.tolist()

    lm = load_multiplets.load_multiplets(GVARS, nl_pruned,
                                         nl_idx_pruned,
                                         omega_pruned)


    # creating the named tuples                                                                                                                                              
    gvars = jf.create_namedtuple('GVAR',
                                 ['r',
                                  'wsr',
                                  's_arr'],
                                 (GVARS_TR.r,
                                  GVARS_TR.wsr,
                                  GVARS_ST.s_arr))    

    # considering self-coupling for testing
    ell1, ell2 = nl_pruned[0,1], nl_pruned[0,1]
    omegaref = omega_pruned[0]
    U1, U2 = lm.U_arr[0], lm.U_arr[0]
    V1, V2 = lm.V_arr[0], lm.V_arr[0]

    qdpt_mode = jf.create_namedtuple('QDPT_MODE',
                                     ['ell1',
                                      'ell2',
                                      'ellmin',
                                      'omegaref'],
                                     (ell1,
                                      ell2,
                                      min(ell1, ell2),
                                      omegaref))
    
    eigfuncs = jf.create_namedtuple('EIGFUNCS',
                                    ['U1', 'U2',
                                     'V1', 'V2'],
                                    (U1, U2,
                                     V1, V2))
    
    wigs = jf.create_namedtuple('WIGNERS',
                                ['wig_list',
                                 'wig_idx_full'],
                                (wig_list,
                                 wig_idx_full))

    Niter = 100

    # creating the instance of the class
    get_submat = compute_submatrix(gvars)


    # testing get_Cvec() function                                                              
    # declaring only qdpt_mode as static argument. It is critical to note that it is
    # better to avoid trying to declare namedtuples containing arrays to be static argument.
    # since for our problem, a changed array will be marked by a changed mode, it is better
    # to club the non-array info in a separate namedtuple than the array info. For example,
    # here, qdpt_mode has non-array info while eigfuncs have array info.
    _get_Cvec = jax.jit(get_submat.jax_get_Cvec(), static_argnums=(0,))
    # __ = _get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref)
    __ = _get_Cvec(qdpt_mode, eigfuncs, wigs)

    
    t1 = time.time()
    for __ in range(Niter): __ = get_submat.jax_get_Cvec()(qdpt_mode, eigfuncs, wigs)
    t2 = time.time()
    

    t3 = time.time()
    for __ in range(Niter): __ = _get_Cvec(qdpt_mode, eigfuncs, wigs).block_until_ready()
    t4 = time.time()
    
    print("get_Cvec()")
    print("JIT version is faster by: ", (t2-t1)/(t4-t3))
    print(f"Time taken per iteration (jax-jitted) get_Cvec = {(t4-t3)/Niter:.3e} seconds")
