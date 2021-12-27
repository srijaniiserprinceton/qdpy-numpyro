import numpy as np
import sys

from qdpy_jax import globalvars as gvar_jax

NAX = np.newaxis

GVARS = gvar_jax.GlobalVars()

# getting the omega0 corresponding to 199 and 200
ind_mult = GVARS.nl_all.index((0,200))

# frequencies in non-dimensional units
omegaref = GVARS.omega_list[ind_mult]

# loading the necessary files
ell1_arr = np.load('ell1_arr.npy')
ell2_arr = np.load('ell2_arr.npy')
cvec_arr = np.load('cvec_arr.npy', allow_pickle=True)


# slicing the Cvec array
def get_Cvec(ell1, ell2):
    index_mask = (np.abs(ell1_arr - ell1) + np.abs(ell2_arr - ell2)) == 0
    Cvec_ell1_ell2 = cvec_arr[index_mask]

    # scaling by 2*omegaref for comparison with supmat_qdpt
    Cvec_ell1_ell2 /= (2 * omegaref)

    return Cvec_ell1_ell2 

ell_nbs = np.array([200, 202, 198, 204, 196], dtype='int')
# ell_nbs += 3
ell0 = ell_nbs[0]

# initializing the supermatrix
dim_super = np.sum(2 * ell_nbs + 1)

dim_submat = np.cumsum(2 * ell_nbs + 1)

# putting the first 0
dim_submat = np.append(np.array([0], dtype='int'),
                        dim_submat)

# making the M as in vorontsov_qdpy
smin_ind, smax_ind = 1, 2
# supermatrix M specific files for the V11 approximated problem                              
param_coeff = np.load('param_coeff.npy')
fixed_part = np.load('fixed_part.npy')
true_params = np.load('true_params.npy')

z = np.sum(param_coeff * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part

M = np.zeros((5, 5, 417))
M_mlen = M.shape[-1]

def make_qdpy_M_shaped():
    for ell1_ind, ell1_true in enumerate(ell_nbs):
        for ell2_ind, ell2_true in enumerate(ell_nbs):
            ell1, ell2 = ell1_true, ell2_true

            print(ell1_ind, ell2_ind, ell1, ell2)

            ellmin = min(ell1, ell2)
            Cvec = get_Cvec(ell1, ell2)[0]
            
            len_cvec = len(Cvec)
            dell = np.abs(len_cvec - M_mlen)//2

            M[ell1_ind, ell2_ind, dell:M_mlen-dell] = Cvec

    return M

M = make_qdpy_M_shaped()

# comparing the M_200
np.testing.assert_array_almost_equal(z[-2], M)
