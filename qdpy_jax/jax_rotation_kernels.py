import numpy as np
import py3nj
import time
import jax
import jax.numpy as jnp


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


def w3j(l1, l2, l3, m1, m2, m3):
    """Computes the wigner-3j symbol for given l1, l2, l3, m1, m2, m3"""
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = int(2*m1)
    m2 = int(2*m2)
    m3 = int(2*m3)
    try:
        wigval = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    except ValueError:
        return 0.0
    return wigval


def w3j_vecm(l1, l2, l3, m1, m2, m3):
    """Computes the wigner-3j symbol for given l1, l2, l3, m1, m2, m3.

    Inputs:
    -------
    l1, l2, l3 - int
    m1, m2, m3 - np.ndarray(ndim=1, dtype=np.int32)

    Returns:
    --------
    wigvals - np.ndarray(ndim=1, dtype=np.float32)
    """
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    return wigvals

def jax_get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref):
    """Computing the non-zero components of the submatrix"""
    # ell = jnp.minimum(ell1, ell2)
    ell = ell1   # !!!!!!!!!!! a temporary fix. This needs to be taken care of
    m = jnp.arange(-ell, ell+1)

    len_s = jnp.size(s_arr)

    wigvals = jnp.zeros((2*ell+1, len_s))

    # for i in range(len_s):
    #    wigvals[:, i] = w3j_vecm(ell1, s_arr[i], ell2, -m, 0*m, m)
     
    wigvals = jax.lax.fori_loop(0, len_s,
                      lambda i, wigvals: jax.ops.index_update(wigvals,jax.ops.index[:,i],1),
                      wigvals)
   
    Tsr = jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2)
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
    integral = jnp.trapz(integrand, axis=1, x=r)
    
    prod_gammas = jax_gamma(ell1) * jax_gamma(ell2) * jax_gamma(s_arr)
    omegaref = omegaref
    Cvec = jax_minus1pow_vec(m) * 8*jnp.pi * omegaref * (wigvals @ (prod_gammas * integral))
    
    return Cvec

def jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
    """Computing the kernels which are used for obtaining the                                                                                                                        
    submatrix elements.                                                                                                                                                              
    """
    Tsr = jnp.zeros((len(s_arr), len(r)))
    
    L1sq = ell1*(ell1+1)
    L2sq = ell2*(ell2+1)
    Om1 = jax_Omega(ell1, 0)
    Om2 = jax_Omega(ell2, 0)
    
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
s_arr = jnp.array([1,3,5], dtype='int')

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
wsr = jnp.asarray(wsr)   # converting to device array once

# using fixed modes (0,200)-(0,200) coupling for testing
n1, n2 = 0, 0
ell1, ell2 = 200, 200

# finding omegaref
omegaref = 1

U = np.loadtxt('U3672.dat')
V = np.loadtxt('V3672.dat')

U = U[rmin_ind:rmax_ind]
V = V[rmin_ind:rmax_ind]

U1, U2 = U, U
V1, V2 = V, V

Niter = 100
'''
# testing the Omega() function
_Omega = jax.jit(jax_Omega)
__ = _Omega(n1,ell1)

t1 = time.time()
for __ in range(Niter): Omega_eval = jax_Omega(n1, ell1).block_until_ready()
t2 = time.time()

t3 = time.time()
for __ in range(Niter): Omega_eval = _Omega(n1, ell1).block_until_ready()
t4 = time.time()

print("Omega()")
print("JIT version is faster by", (t2-t1)/(t4-t3)) 


# testing the minus1pow() function
_minus1pow = jax.jit(jax_minus1pow)
__ = _minus1pow(29)

t1 = time.time()
for __ in range(Niter): minus1pow_eval = jax_minus1pow(29).block_until_ready()
t2 = time.time()


t3 = time.time()
for __ in range(Niter): minus1pow_eval = _minus1pow(29).block_until_ready()
t4 = time.time()

print("minus1pow()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3)) 


# testing the minus1pow_vec() function
_minus1pow_vec = jax.jit(jax_minus1pow_vec)
__ = _minus1pow_vec(jnp.arange(29))

t1 = time.time()
for __ in range(Niter): minus1pow_vec_eval = jax_minus1pow_vec(jnp.arange(29)).block_until_ready()
t2 = time.time()


t3 = time.time()
for __ in range(Niter): minus1pow_vec_eval = _minus1pow_vec(jnp.arange(29)).block_until_ready()
t4 = time.time()

print("minus1pow_vec()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3)) 


# testing the gamma() function
_gamma = jax.jit(jax_gamma)
__ = _gamma(29)

t1 = time.time()
for __ in range(Niter): minus1pow_eval = jax_gamma(29).block_until_ready()
t2 = time.time()


t3 = time.time()
for __ in range(Niter): minus1pow_eval = _gamma(29).block_until_ready()
t4 = time.time()

print("gamma()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3)) 
'''


# testing compute_Tsr() function
_compute_Tsr = jax.jit(jax_compute_Tsr)
__ = _compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2)

t1 = time.time()
for __ in range(Niter): __ = jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2).block_until_ready()
t2 = time.time()

t3 = time.time()
for __ in range(Niter): __ = _compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2).block_until_ready()
t4 = time.time()

print("compute_Tsr()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3)) 
print(f"Time taken per iteration (jax jitted) compute_Tsr = {(t4-t3)/Niter:.3e} seconds")


# testing get_Cvec() function                                                                                                                                                                           
_get_Cvec = jax.jit(jax_get_Cvec, static_argnums=(0,1))
__ = _get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref)

t1 = time.time()
for __ in range(Niter): __ = jax_get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref).block_until_ready()
t2 = time.time()

t3 = time.time()
for __ in range(Niter): __ = _get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref).block_until_ready()
t4 = time.time()

print("get_Cvec()")
print("JIT version is faster by: ", (t2-t1)/(t4-t3))
print(f"Time taken per iteration (jax jitted) get_Cvec = {(t4-t3)/Niter:.3e} seconds")
