Bimport numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import globalvars as gvar_jax
from vorontsov_qdpy import sparse_precompute_bkm as precompute_bkm

'''For this test file, we need to set n=0 and l=199 to 204
even in vorontsov_qdpy.'''

ARGS = np.loadtxt(".n0-lmin-lmax.dat")
GVARS = gvar_jax.GlobalVars(n0=int(ARGS[0]),
                            lmin=int(ARGS[1]),
                            lmax=int(ARGS[2]),
                            rth=ARGS[3],
                            knot_num=int(ARGS[4]),
                            load_from_file=int(ARGS[5]))

NAX = np.newaxis

def plot_compare_bkm(bkm_jax):
    # comparing with the bkm from Woordard13
    bkm_W13_dpt = np.load('bkm_dpt.npy')
    
    # plotting
    fig, ax = plt.subplots(2, 3, figsize=(10,6))
    
    # ells to be plotted
    ell_plot = np.array([204, 203, 202, 201, 200, 199])
    
    ell_count = 0
    
    for row in range(2):
        for col in range(3):
            ell = ell_plot[ell_count]
            
            ax[row,col].plot(bkm_jax[ell_count, 0, max_lmax:-(max_lmax-(ell-1))], 'r', label='bkm_jax')
            ax[row,col].plot(bkm_W13_dpt[ell-194, 0, 6:-(215-6-ell)], '--k', label='bkm_W13_dpt')
            
            ell_count += 1
        
    plt.legend()
    
    plt.savefig('compare_bkm.png')
    
    plt.close()

def compute_bkm_scaled():
    # computing the bkm from the .n0_lmin_lmax file in tests
    # currently it is n = 0 , l = 199 to 204
    noc_bkm, fixed_bkm, k_arr, p_arr = precompute_bkm.build_bkm_all_cenmults()
    
    # loading true_params
    true_params = GVARS.ctrl_arr_dpt_clipped
    
    # constructing bkm full
    bkm = np.sum(noc_bkm * true_params[NAX, :, :, NAX, NAX], axis=(1,2)) + fixed_bkm
    dom_dell = GVARS.dom_dell
    
    bkm_scaled = -1.0 * bkm / dom_dell[:, NAX, NAX]

    # loading precomputed benchmarked value
    bkm_test = np.load('bkm_test.npy')

    # testing against a benchmarked values stored
    np.testing.assert_array_almost_equal(bkm_scaled, bkm_test)

    return bkm_scaled

if __name__=="__main__":
    bkm_scaled = compute_bkm_scaled() 
    
    # necessary shapes for making arrays                                                      
    max_lmax, max_nbs = precompute_bkm.get_max_lmax_and_nbs()

    plot_compare_bkm(bkm_scaled)
