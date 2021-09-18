import jax.numpy as jnp
import numpy as np
import jax
import py3nj
import time

jit = jax.jit

def Omega(ell, N):
    """Computes Omega_N^\ell"""
    if abs(N) > ell:
        return 0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1))

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



def minus1pow(num):
    """Computes (-1)^n"""
    if num%2 == 1:
        return -1.0
    else:
        return 1.0

def pure_uses_internal_state(x):
    state = dict(even=0, odd=0)
    for i in range(10):
        state['even' if i % 2 == 0 else 'odd'] += x
    return state['even'] + state['odd']


def compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
    """Computing the kernels which are used for obtaining the                                                                                                                        
    submatrix elements.                                                                                                                                                              
    """
    
    Tsr = np.zeros((len(s_arr), len(r)))
    
    L1sq = ell1*(ell1+1)
    L2sq = ell2*(ell2+1)
    Om1 = Omega(ell1, 0)
    Om2 = Omega(ell2, 0)
    
    for i in range(len(s_arr)):
        s = s_arr[i]
        ls2fac = L1sq + L2sq - s*(s+1)
        eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
        wigval = w3j(ell1, s, ell2, -1, 0, 1)
        Tsr[i, :] = -(1 - minus1pow(ell1 + ell2 + s)) * \
                    Om1 * Om2 * wigval * eigfac / r
        
    return Tsr

Niter = 1000

t1 = time.time()
for i in range(Niter): __ = pure_uses_internal_state(5.)
t2 = time.time()



t3 = time.time()
for i in range(Niter): __ = jit(pure_uses_internal_state)(5.)
t4 = time.time()

print((t2-t1)/Niter)
print((t4-t3)/Niter)

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

# jitting computation of Tsr kernel. Maybe use a leading\
# underscore to indicate jitted functions?
_compute_Tsr = jit(compute_Tsr)

Tsr = compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2) #.block_until_ready()
