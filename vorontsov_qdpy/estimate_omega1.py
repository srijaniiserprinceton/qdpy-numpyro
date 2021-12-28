import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import globalvars as gvar_jax

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

NAX = np.newaxis

# loading the true parameters
true_params = np.load('true_params.npy')

# loading the exact supmat files
fixed_part = np.load('fixed_part.npy')
param_coeff = np.load('param_coeff.npy')
freq_diag = np.load('freq_diag.npy')

# loading the M files
fixed_part_M = np.load('fixed_part_M.npy')
param_coeff_M = np.load('param_coeff_M.npy')
p_dom_dell = np.load('p_dom_dell.npy')

# generating the exact supmat Z
Z = np.sum(param_coeff * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part + freq_diag

# generating the 0th order Z
Z0 = np.sum(param_coeff_M * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part_M + p_dom_dell

# generating the 1st order Z
Z1 = Z - Z0

def plot_compare():
    for i in range(Z.shape[0]):
        fig, ax = plt.subplots(5,5,figsize=(10,10))
        count = 0
        for row in range(5):
            for col in range(5):
                ax[row,col].plot(Z[i,row,col], 'r', alpha=0.5)
                ax[row,col].plot(Z0[i,row,col], 'k', alpha=0.5)
                ax[row,col].plot(Z1[i,row,col], 'b', alpha=0.5)
                count += 1

        plt.savefig(f'Z_compare_{i}.png')

        print(i)

# plotting to compare the Z and Z0 and Z1 values
plot_compare()

# now building the clp to get the eigenfunction corrections
param_coeff_bkm = np.load('noc_bkm.npy')
fixed_bkm_sparse = np.load('fixed_bkm.npy')
true_params = np.load('true_params.npy')
k_arr = np.load('k_arr.npy')
p_arr = np.load('p_arr.npy')

#------------------------intermediate bkm test---------------------#                          
#----------do this only for n=0, l between 194 and 208-------------#                         
# constructing bkm full                                                                      
bkm_jax = np.sum(param_coeff_bkm * true_params[:, :, NAX, NAX, NAX], axis=(0,1)) \
          + fixed_bkm_sparse
dom_dell = GVARS.dom_dell
bkm_scaled = -1.0 * bkm_jax / dom_dell[:, NAX, NAX]

# loading precomputed benchmarked value                                                      
bkm_test = np.load('../tests/bkm_test.npy')

# testing against a benchmarked values stored                                                 
np.testing.assert_array_almost_equal(bkm_scaled, bkm_test)
        
# function to build clp
def get_clp(bkm):
    k_arr_denom = k_arr * 1
    k_arr_denom[k_arr==0] = np.inf

    tvals = np.linspace(0, np.pi, 25)

    # integrand of shape (ell, p, m ,t)                                                       
    integrand = np.zeros((p_arr.shape[0],
                          p_arr.shape[1],
                          p_arr.shape[2],
                          len(tvals)))

    for i in range(len(tvals)):
        term2 = 2*bkm*np.sin(k_arr*tvals[i])/k_arr_denom
        term2 = term2.sum(axis=1)
        integrand[:,:,:,i] = np.cos(p_arr*tvals[i] - term2[:, NAX, :])

    integral = np.trapz(integrand, axis=-1, x=tvals)/np.pi
    return integral

clp = get_clp(bkm_scaled)


# computing omega1
omega1 = np.sum(clp * np.sum(Z1 * clp[:,:,NAX,:], axis=1), axis=1)
