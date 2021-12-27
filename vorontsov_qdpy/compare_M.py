import numpy as np
import sys

from qdpy_jax import globalvars as gvar_jax

NAX = np.newaxis

GVARS = gvar_jax.GlobalVars()

# getting the omega0 corresponding to 199 and 200
ind_202 = GVARS.nl_all.index((0,202))
ind_203 = GVARS.nl_all.index((0,203))

# frequencies in non-dimensional units
omega_202 = GVARS.omega_list[ind_202]
omega_203 = GVARS.omega_list[ind_203]

# loading the necessary files
ell1_arr = np.load('ell1_arr_203.npy')
ell2_arr = np.load('ell2_arr_203.npy')
cvec_arr = np.load('cvec_arr_203.npy', allow_pickle=True)


# slicing the Cvec array
def get_Cvec(ell1, ell2):
    index_mask = (np.abs(ell1_arr - ell1) + np.abs(ell2_arr - ell2)) == 0
    Cvec_ell1_ell2 = cvec_arr[index_mask]

    # scaling by 2*omegaref for comparison with supmat_qdpt
    if(ell1 % 2 == 0): Cvec_ell1_ell2 /= (2 * omega_202)
    else: Cvec_ell1_ell2 /= (2 * omega_203)

    return Cvec_ell1_ell2 

ell_nbs = np.array([200, 202, 198, 204, 196], dtype='int')
ell_nbs += 3
ell0 = ell_nbs[0]

# initializing the supermatrix
dim_super = np.sum(2 * ell_nbs + 1)
supermatrix = np.zeros((dim_super, dim_super))

dim_submat = np.cumsum(2 * ell_nbs + 1)

# putting the first 0
dim_submat = np.append(np.array([0], dtype='int'),
                        dim_submat)

for ell1_ind, ell1_true in enumerate(ell_nbs):
    for ell2_ind, ell2_true in enumerate(ell_nbs):
        k = (ell1_true - ell2_true) // 2
        ell1, ell2 = ell0 + k, ell0 - k

        ellmin = min(ell1, ell2)
        ellmin_true = min(ell1_true, ell2_true)
        Cvec = get_Cvec(ell1, ell2)[0]

        # cropping off Cvec if length larger than ellmin_true
        if(ellmin_true < ellmin):
            dellmin = np.abs(ellmin - ellmin_true)
            Cvec = Cvec[dellmin:-dellmin]

        startx = dim_submat[ell1_ind]
        starty = dim_submat[ell2_ind]

        # the length of padding
        ellmin_pad = min(ellmin, ellmin_true)
        dellx = np.abs(ell1_true - ellmin_pad)
        delly = np.abs(ell2_true - ellmin_pad)

        # length of Cvec
        len_cvec = len(Cvec)

        # tiling in Cvec
        supermatrix[startx+dellx:startx+dellx+len_cvec,
                    starty+delly:starty+delly+len_cvec] = np.diag(Cvec)

        
# making the M as in vorontsov_qdpy
smin_ind, smax_ind = 1, 2
# supermatrix M specific files for the V11 approximated problem                              
param_coeff_M = np.load('param_coeff_M.npy')
sparse_idx_M = np.load('sparse_idx_M.npy')
fixed_part_M = np.load('fixed_part_M.npy')
true_params = np.load('true_params.npy')

z0 = np.sum(param_coeff_M * true_params[:,:,NAX,NAX,NAX,NAX], axis=(0,1)) \
     + fixed_part_M

M = np.zeros((5, 5, 417))
M_mlen = M.shape[-1]

def make_qdpy_M_shaped():
    for ell1_ind, ell1_true in enumerate(ell_nbs):
        for ell2_ind, ell2_true in enumerate(ell_nbs):
            k = (ell1_true - ell2_true) // 2
            ell1, ell2 = ell0 + k, ell0 - k
            
            print(ell1_ind, ell2_ind, ell1_true, ell2_true, ell1, ell2)

            ellmin = min(ell1, ell2)
            ellmin_true = min(ell1_true, ell2_true)
            Cvec = get_Cvec(ell1, ell2)[0]
            
            # cropping off Cvec if length larger than ellmin_true                           
            if(ellmin_true < ellmin):
                dellmin = np.abs(ellmin - ellmin_true)
                Cvec = Cvec[dellmin:-dellmin]

            len_cvec = len(Cvec)
            dell = np.abs(len_cvec - M_mlen)//2

            M[ell1_ind, ell2_ind, dell:M_mlen-dell] = Cvec


    return M

M = make_qdpy_M_shaped()

# comparing the M_200
np.testing.assert_array_almost_equal(z0[-5], M)
