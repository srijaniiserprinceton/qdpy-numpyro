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

def jax_compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
    """Computing the kernels which are used for obtaining the                                                                                                                        
    submatrix elements.                                                                                                                                                              
    """
    
    Tsr = jnp.zeros((len(s_arr), len(r)))
    
    L1sq = ell1*(ell1+1)
    L2sq = ell2*(ell2+1)
    Om1 = jax_Omega(ell1, 0)
    Om2 = jax_Omega(ell2, 0)
    
    for i in range(len(s_arr)):
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


# parameters to be included in the global dictionary later?
s_arr = np.array([1,3,5], dtype='int')

rmin = 0.3
rmax = 1.0

r = np.loadtxt('r.dat') # the radial grid

# finding the indices for rmin and rmax
rmin_ind = np.argmin(np.abs(r - rmin))
rmax_ind = np.argmin(np.abs(r - rmax)) + 1

# clipping radial grid
r = r[rmin_ind:rmax_ind]

# using fixed modes (0,200)-(0,200) coupling for testing
n1, n2 = 0, 0
ell1, ell2 = 200, 200

U = np.loadtxt('U3672.dat')
V = np.loadtxt('V3672.dat')

U = U[rmin_ind:rmax_ind]
V = V[rmin_ind:rmax_ind]

U1, U2 = U, U
V1, V2 = V, V

Niter = 1000

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

