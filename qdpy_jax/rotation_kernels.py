import numpy as np
import py3nj
import time

def gamma(ell):
    """Computes gamma_ell"""
    return np.sqrt((2*ell + 1)/4/np.pi)



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


def minus1pow_vec(num):
    """Computes (-1)^n"""
    mask_pos = num%2 == 0
    ret = np.zeros_like(num, dtype=np.int32)
    ret[mask_pos] = 1
    ret[~mask_pos] = -1
    return ret


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


def get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref):
    """Computing the non-zero components of the submatrix"""
    # ell = np.minimum(ell1, ell2)
    ell = ell1   # !!!!!!!!!!! a temporary fix. This needs to be taken care of
    m = np.arange(-ell, ell+1)

    len_s = np.size(s_arr)

    wigvals = np.zeros((2*ell+1, len_s))

    for i in range(len_s):
        wigvals[:, i] = w3j_vecm(ell1, s_arr[i], ell2, -m, 0*m, m)
     
    Tsr = compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2)
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
    integral = np.trapz(integrand, axis=1, x=r)
    
    prod_gammas = gamma(ell1) * gamma(ell2) * gamma(s_arr)
    omegaref = omegaref
    Cvec = minus1pow_vec(m) * 8*np.pi * omegaref * (wigvals @ (prod_gammas * integral))
    
    return Cvec



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

wsr = np.loadtxt('w.dat')
wsr = wsr[:,rmin_ind:rmax_ind]

U1, U2 = U, U
V1, V2 = V, V

omegaref = 1234.

Niter = 100

t1 = time.time()
for __ in range(Niter): Tsr = compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2)
t2 = time.time()

print(f"[compute_Tsr] Time taken per iteration in seconds (no jax): , {(t2-t1):.3e} seconds")



t3 = time.time()
for __ in range(Niter): __ = get_Cvec(ell1, ell2, s_arr, r, U1, U2, V1, V2, omegaref)
t4 = time.time()
print(f"[get_Cvec] Time taken per iteration in seconds (no jax): , {(t4-t3):.3e} seconds")
