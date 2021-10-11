from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from jax import random, vmap
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import numpy as np
import os

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import handlers

from qdpy_jax import rotation_kernels as rkern

NAX = np.newaxis

plt.style.use('bmh')
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats('svg')

numpyro.set_platform("cpu")
#numpyro.set_host_device_count(5)

from scipy.integrate import simps
from jax.numpy import trapz



def eig_mcmc_func(w1=None, w3=None, w5=None):
    return  get_eigs(create_supermatrix(w1, w3, w5))/2./omega0
    

def create_supermatrix(w1, w3, w5):
    integrand1 = Tsr[0, :] * w1
    integrand3 = Tsr[1, :] * w3
    integrand5 = Tsr[2, :] * w5
    integral1 = trapz(integrand1, x=r)
    integral3 = trapz(integrand3, x=r)
    integral5 = trapz(integrand5, x=r)
    prod_gamma1 = gamma(ell1)*gamma(ell2)*gamma(s_arr[0])
    prod_gamma3 = gamma(ell1)*gamma(ell2)*gamma(s_arr[0])
    prod_gamma5 = gamma(ell1)*gamma(ell2)*gamma(s_arr[0])
    wpi = (wigvals[:, 0]*integral1*prod_gamma1 + 
           wigvals[:, 1]*integral3*prod_gamma3 +
           wigvals[:, 2]*integral5*prod_gamma5)
    diag = 8*np.pi*omega0*wpi
    supmat = jnp.diag(diag)
    return supmat

def get_eigs(mat):
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    return eigvals

def gamma(ell):
    return jnp.sqrt((2*ell+1)/4./np.pi)

def minus1pow_vecm(num):
    modval = num % 2
    retval = np.zeros_like(modval)
    retval[modval == 1] = -1
    retval[modval == 0] = 1
    return retval

def model():
    # setting min and max value to be 0.1*true and 3.*true
    w1min, w1max = .1*abs(w1t), 3.*abs(w1t)
    w3min, w3max = .1*abs(w3t), 3.*abs(w3t)
    w5min, w5max = .1*abs(w5t), 3.*abs(w5t)
    
    w1 = numpyro.sample('w1', dist.Uniform(w1min, w1max))
    w3 = numpyro.sample('w3', dist.Uniform(w3min, w3max))
    w5 = numpyro.sample('w5', dist.Uniform(w5min, w5max))
    
    sigma = numpyro.sample('sigma', dist.Uniform(0.1, 10.0))
    eig_sample = numpyro.deterministic('eig', eig_mcmc_func(w1=w1, w3=w3, w5=w5))
    return numpyro.sample('obs', dist.Normal(eig_sample, sigma), obs=eigvals_true)



if __name__ == "__main__":
    # parameters to be included in the global dictionary later?
    s_arr = np.array([1,3,5], dtype='int')

    rmin = 0.3
    rmax = 1.0

    r = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/r.dat') # the radial grid

    # finding the indices for rmin and rmax
    rmin_ind = np.argmin(np.abs(r - rmin))
    rmax_ind = np.argmin(np.abs(r - rmax)) + 1

    # clipping radial grid
    r = r[rmin_ind:rmax_ind]

    # using fixed modes (0,200)-(0,200) coupling for testing
    n1, n2 = 0, 0
    ell1, ell2 = 200, 200

    U = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/U3672.dat')
    V = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/V3672.dat')

    U = U[rmin_ind:rmax_ind]
    V = V[rmin_ind:rmax_ind]

    wsr = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/w.dat')
    wsr = wsr[:,rmin_ind:rmax_ind]

    U1, U2 = U, U
    V1, V2 = V, V

    omegaref = 1234.

    Niter = 100
    Tsr = rkern.compute_Tsr(ell1, ell2, s_arr, r,
                            U1, U2, V1, V2)

    r = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/r.dat')
    U = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/U3672.dat')
    V = np.loadtxt('/mnt/disk2/samarth/qdpy-numpyro/qdpy_jax/V3672.dat')

    rmin, rmax = 0.3, 1.0
    rmin_idx = np.argmin(abs(r - rmin))
    rmax_idx = np.argmin(abs(r - rmax)) + 1

    r = r[rmin_idx:rmax_idx]
    U = U[rmin_idx:rmax_idx]
    V = V[rmin_idx:rmax_idx]

    r = jnp.asarray(r)
    U = jnp.asarray(U)
    V = jnp.asarray(V)
    Tsr = jnp.asarray(Tsr)

    n1, n2 = 0, 0
    ell1, ell2 = 200, 200
    ell = min(ell1, ell2)
    nu0 = 4741.

    Msol = 1.989e33
    Rsol = 6.956e10
    B0 = 10.e5
    OM = jnp.sqrt(4*np.pi*Rsol*B0**2/Msol)
    nu0 /= OM*1e6
    omega0 = 2*jnp.pi*nu0

    s_arr = jnp.array([1., 3., 5.])
    m = jnp.arange(-ell, ell+1)
    wigvals = np.ones((2*ell+1, len(s_arr)))
    # for i in range(len(s_arr)):
        # wigvals[:, i] = w3j_vecm(ell1, s_arr[i], ell2, -m, 0*m, m)

    wigvals = jnp.asarray(wigvals)
    w1t, w3t, w5t = 10.51, .43, .175
    eigvals_true = get_eigs(create_supermatrix(w1t, w3t, w5t))/2/omega0

    rng_key = random.PRNGKey(12)
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS.
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=50, num_samples=1000)#, num_chains=5)

    mcmc.run(rng_key_)
    #mcmc.run(rng_key_, x_scaled=x_scaled)
    mcmc.print_summary()
    print(f"w1_true = {w1t}\n" +
        f"w3_true = {w3t}\n" +
        f"w5_true = {w5t}")
