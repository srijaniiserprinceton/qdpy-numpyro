import numpy as np
import time
import jax
import jax.numpy as jnp
from collections import namedtuple
import sys

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


#def jax_get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref):
def jax_get_Cvec(gvars, qdpt_mode, eigfuncs):
    """Computing the non-zero components of the submatrix"""
    # ell = jnp.minimum(ell1, ell2)
    ell = qdpt_mode.ell1   # !!!!!!!!!!! a temporary fix. This needs to be taken care of
    m = jnp.arange(-ell, ell+1)

    len_s = jnp.size(gvars.s_arr)

    wigvals = jnp.zeros((2*ell+1, len_s))

    # for i in range(len_s):
    #    wigvals[:, i] = w3j_vecm(ell1, s_arr[i], ell2, -m, 0*m, m)
     
    jax.lax.fori_loop(0, len_s,
                      lambda i, wigvals: jax.ops.index_update(wigvals,jax.ops.index[:,i],1),
                      wigvals)
   
    # Tsr = jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2)
    Tsr = jax_compute_Tsr(gvars, qdpt_mode, eigfuncs)
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
    
    integrand = Tsr * wsr   # since U and V are scaled by sqrt(rho) * r                                                                                             
    
    #### TO BE REPLACED WITH SIMPSON #####
    integral = jnp.trapz(integrand, axis=1, x=gvars.r)
    
    prod_gammas = jax_gamma(qdpt_mode.ell1) * jax_gamma(qdpt_mode.ell2) * jax_gamma(gvars.s_arr)
    omegaref = qdpt_mode.omegaref
    Cvec = jax_minus1pow_vec(m) * 8*jnp.pi * qdpt_mode.omegaref * (wigvals @ (prod_gammas * integral))
    
    return Cvec

#def jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
def jax_compute_Tsr(gvars, qdpt_mode, eigfuncs): 
    """Computing the kernels which are used for obtaining the                                                                                                    
    submatrix elements.                                                                                                                                               
    """
    Tsr = jnp.zeros((len(gvars.s_arr), len(gvars.r)))
    
    L1sq = qdpt_mode.ell1*(qdpt_mode.ell1+1)
    L2sq = qdpt_mode.ell2*(qdpt_mode.ell2+1)
    Om1 = jax_Omega(qdpt_mode.ell1, 0)
    Om2 = jax_Omega(qdpt_mode.ell2, 0)
    
    U1, U2, V1, V2 = eigfuncs.U1, eigfuncs.U2, eigfuncs.V1, eigfuncs.V2

    # creating internal function for the fori_loop                                                     
    def func4Tsr_s_loop(i, Tsr):
        s = s_arr[i]
        ls2fac = L1sq + L2sq - s*(s+1)
        eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
        # wigval = w3j(ell1, s, ell2, -1, 0, 1)                                                       
        # using some dummy number until we write the                                                  
        # function for mapping wigner3js                                                              
        wigval = 1.0
        Tsr_at_i = -(1 - jax_minus1pow(ell1 + ell2 + s)) * \
                    Om1 * Om2 * wigval * eigfac / r
        Tsr = jax.ops.index_update(Tsr, i, Tsr_at_i)

        return Tsr
    
    Tsr = jax.lax.fori_loop(0, len(s_arr), func4Tsr_s_loop, Tsr)

    return Tsr

# parameters to be included in the global dictionary later?
s_arr = jnp.array([1,3,5], dtype='int32')

rmin = 0.3
rmax = 1.0

r = np.loadtxt('r.dat') # the radial grid

# finding the indices for rmin and rmax
rmin_ind = np.argmin(np.abs(r - rmin))
rmax_ind = np.argmin(np.abs(r - rmax)) + 1

# clipping radial grid
r = r[rmin_ind:rmax_ind]

# the rotation profile
wsr = np.loadtxt('w.dat')
wsr = wsr[:,rmin_ind:rmax_ind]
wsr = jnp.array(wsr)   # converting to device array once

# using fixed modes (0,200)-(0,200) coupling for testing
n1, n2 = 0, 0
ell1, ell2 = 200, 200

# finding omegaref
omegaref = 1

U = np.loadtxt('U3672.dat')
V = np.loadtxt('V3672.dat')

U = U[rmin_ind:rmax_ind]
V = V[rmin_ind:rmax_ind]

# converting numpy arrays to jax.numpy arrays
r = jnp.array(r)
U, V = jnp.array(U), jnp.array(V)

U1, U2 = U, U
V1, V2 = V, V

# creating the named tuples
GVAR = namedtuple('GVAR', 'r wsr s_arr')
QDPT_MODE = namedtuple('QDPT_MODE', 'ell1 ell2 omegaref')
EIGFUNCS = namedtuple('EIGFUNCS', 'U1 U2 V1 V2')

# initializing namedtuples. This could be done from a separate file later
gvars = GVAR(r, wsr, s_arr)
qdpt_mode = QDPT_MODE(ell1, ell2, omegaref)
eigfuncs = EIGFUNCS(U1, U2, V1, V2)

Niter = 100

# testing get_Cvec() function                                                                                                                                                           # declaring only qdpt_mode as static argument. It is critical to note that it is
# better to avoid trying to declare namedtuples containing arrays to be static argument.
# since for our problem, a changed array will be marked by a changed mode, it is better
# to club the non-array info in a separate namedtuple than the array info. For example,
# here, qdpt_mode has non-array info while eigfuncs have array info.
_get_Cvec = jax.jit(jax_get_Cvec, static_argnums=(1,))
# __ = _get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref)
__ = _get_Cvec(gvars, qdpt_mode, eigfuncs)

t1 = time.time()
for __ in range(Niter): __ = jax_get_Cvec(gvars, qdpt_mode, eigfuncs).block_until_ready()
t2 = time.time()

t3 = time.time()
for __ in range(Niter): __ = _get_Cvec(gvars, qdpt_mode, eigfuncs).block_until_ready()
t4 = time.time()

print("get_Cvec()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3))
print(f"Time taken per iteration (jax-jitted) get_Cvec = {(t4-t3)/Niter:.3e} seconds")
