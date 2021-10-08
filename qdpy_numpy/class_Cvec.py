import numpy as np
import time
from collections import namedtuple
from scipy.integrate import trapz
import py3nj
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
data_dir = f"{package_dir}/qdpy_jax"



def Omega(ell, N):
    """Computes Omega_N^\ell"""
    if abs(N) > ell:
        return 0.0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1)),
   
def minus1pow(num):
    """Computes (-1)^n"""
    if num % 2 == 1:
        return -1
    else:
        return 1

def minus1pow_vec(num):
    """Computes (-1)^n"""
    modval = num % 2
    return (-1)**modval

def gamma(ell):
    """Computes gamma_ell"""
    return np.sqrt((2*ell + 1)/4/np.pi)


def w3j_vecm(l1, l2, l3, m1, m2, m3):
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    return wigvals

def w3j(l1, l2, l3, m1, m2, m3):
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




class compute_submatrix:
    def __init__(self, gvars):
        self.r = gvars.r
        self.s_arr = gvars.s_arr
        self.wsr = gvars.wsr
        
    def get_Cvec(self, qdpt_mode, eigfuncs):
        """Computing the non-zero components of the submatrix"""
        # ell = jnp.minimum(ell1, ell2)
        ell = qdpt_mode.ell1   # !!!!!!!!!!! a temporary fix. This needs to be taken care of
        m = np.arange(-ell, ell+1)

        len_s = np.size(self.s_arr)

        wigvals = np.zeros((2*ell+1, len_s))

        for i in range(len_s):
            wigvals[:, i] = w3j_vecm(ell1, s_arr[i], ell2, -m, 0*m, m)

        Tsr = self.compute_Tsr(qdpt_mode, eigfuncs)
        integrand = Tsr * self.wsr   # since U and V are scaled by sqrt(rho) * r                                                                                             

        #### TO BE REPLACED WITH SIMPSON #####
        integral = trapz(integrand, axis=1, x=self.r)

        prod_gammas = gamma(qdpt_mode.ell1) * gamma(qdpt_mode.ell2) * gamma(self.s_arr)
        omegaref = qdpt_mode.omegaref
        Cvec = minus1pow_vec(m) * 8*np.pi * qdpt_mode.omegaref * (wigvals @ (prod_gammas * integral))

        return Cvec

    #def compute_Tsr(ell1, ell2, s_arr, r, U1, U2, V1, V2):
    def compute_Tsr(self, qdpt_mode, eigfuncs): 
        """Computing the kernels which are used for obtaining the                                                                                                    
        submatrix elements.                                                                                                                                               
        """
        Tsr = np.zeros((len(self.s_arr), len(self.r)))
        
        L1sq = qdpt_mode.ell1*(qdpt_mode.ell1+1)
        L2sq = qdpt_mode.ell2*(qdpt_mode.ell2+1)
        Om1 = Omega(qdpt_mode.ell1, 0)[0]
        Om2 = Omega(qdpt_mode.ell2, 0)[0]
        
        U1, U2, V1, V2 = eigfuncs.U1, eigfuncs.U2, eigfuncs.V1, eigfuncs.V2
        
        # creating internal function for the fori_loop

        for i in range(len(self.s_arr)):
            s = self.s_arr[i]
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            wigval = 1.0 #using dummy for now
            Tsr[i] = -(1. - minus1pow(int(ell1 + ell2 + s))) * \
                       Om1 * Om2 * wigval * eigfac / r

        return Tsr

# parameters to be included in the global dictionary later?
s_arr = np.array([1,3,5], dtype='int32')

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
wsr = np.array(wsr)   # converting to device array once

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
r = np.array(r)
U, V = np.array(U), np.array(V)

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

# creating the instance of the class
get_submat = compute_submatrix(gvars)
t3 = time.time()
for __ in range(Niter): __ = get_submat.get_Cvec(qdpt_mode, eigfuncs)
t4 = time.time()

print(f"Time taken per iteration (numpy) get_Cvec = {(t4-t3)/Niter:.3e} seconds")
