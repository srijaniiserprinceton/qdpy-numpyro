import jax.numpy as jnp
import jax
import time

jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform) 

# function to check if the elements of a 1D array are sorted
def issorted(a):
    return jnp.all(a[:-1] <= a[1:])

# function to find 2-d index for a contiguous numbering
# of elements in a matrix
def ind2sub(cont_ind, nrows, ncols):
    return cont_ind//nrows, cont_ind%ncols

def find_c_RY03(ell1, ell2, ell3, m1, m2, m3):
    # following Rasch and Yu, 2003 (RY03)
    # /ell1 ell2 ell3\
    # \ m1   m2   m3 /
    
    # forming the R matrix according to Eqn.(2.10) in RY03
    R = jnp.array([[-ell1 + ell2 + ell3, ell1 - ell2 + ell3, ell1 + ell2 - ell3],
                   [ell1 - m1, ell2 - m2, ell3 - m3],
                   [ell1 + m1, ell2 + m2, ell3 + m3]]) 
    
    # converting it to the Eqn.(2.11) form according to the step in
    # ~https://github.com/csdms-contrib/slepian_alpha/blob/master/wignersort.m~

    # to store various indices during the operation
    # temp = jnp.array([1,2], dtype='int32')
    temp = jnp.zeros(2, dtype='int32')
    # to store information of the phase re-adjustment needed
    oddperm = jnp.array([0], dtype='int32')

    # finding the location of the smallest elements S
    S_cont_index = jnp.argmin(R)
    S_row_ind, S_col_ind = ind2sub(S_cont_index, 3, 3)
    temp = jax.ops.index_update(temp, 0, S_row_ind)
    temp = jax.ops.index_update(temp, 1, S_col_ind)
    
    # moving S to the (0,0) position
    R = jnp.roll(R, -temp[0], axis=0)
    R = jnp.roll(R, -temp[1], axis=1)

    # finding the location of the largest element L
    L_cont_ind = jnp.argmax(R)
    L_row_ind, L_col_ind = ind2sub(L_cont_ind, 3, 3)
    temp = jax.ops.index_update(temp, 0, L_row_ind)
    temp = jax.ops.index_update(temp, 1, L_col_ind)
    
    #Reorder Regge square
    # in this part there are a bunch of inequalities
    # that need to be checked to carry out row and 
    # col swapping operations. Defining the miscellaneous
    # internal functions for jax.lax.cond

    def true_func_1(R_and_oddperm):
        R, oddperm = R_and_oddperm
        R = jnp.transpose(R)
        def true_func_1_true_func(R_and_oddperm):
            R, oddperm = R_and_oddperm
            R = jax.ops.index_update(R,jax.ops.index[:,1:3],jnp.fliplr(R[:,1:3]))
            oddperm = jax.ops.index_add(oddperm,0,1)
            return R, oddperm
            
        R_and_oddperm = jax.lax.cond(temp[0] == 2,
                                     true_func_1_true_func,
                                     lambda R_and_oddperm: R_and_oddperm,
                                     operand=(R, oddperm))
        
        return R_and_oddperm

    def false_func_1(R_and_oddperm):
        def false_func_1_true_func(R_and_oddperm):
            R, oddperm = R_and_oddperm
            R = jax.ops.index_update(R,jax.ops.index[:,1:3],jnp.fliplr(R[:,1:3]))
            oddperm = jax.ops.index_add(oddperm,0,1)
            return R, oddperm

        R_and_oddperm = jax.lax.cond(temp[1] == 2,
                                     false_func_1_true_func,
                                     lambda R_and_oddperm: R_and_oddperm,
                                     operand=R_and_oddperm)
        
        return R_and_oddperm

    def true_func_2(R_and_oddperm):
        R, oddperm = R_and_oddperm
        oddperm = jax.ops.index_update(oddperm, 0, 1-oddperm[0])
        R = jax.ops.index_update(R,jax.ops.index[1:3,:],jnp.flipud(R[1:3,:]))
        return (R, oddperm)

    def false_func_2(R_and_oddperm):
        R, oddperm = R_and_oddperm
        def false_func_2_true_func(R_and_oddperm):
            R, oddperm = R_and_oddperm
            R = jax.ops.index_update(R,jax.ops.index[1:3,:],jnp.flipud(R[1:3,:]))
            oddperm = jax.ops.index_update(oddperm, 0, 1-oddperm[0])
            return (R, oddperm)
            
        R_and_oddperm = jax.lax.cond((R[2,1] == R[1,1])*(R[2,2] < R[1,2]),
                     false_func_2_true_func,
                     lambda R_and_oddperm: R_and_oddperm,
                     operand=R_and_oddperm)

        return R_and_oddperm

    R, oddperm = jax.lax.cond(temp[1] == 0,
                              true_func_1,
                              false_func_1,
                              operand=(R, oddperm))

    R, oddperm = jax.lax.cond(R[2,1] < R[1,1],
                              true_func_2,
                              false_func_2,
                              operand=(R, oddperm))
    
    # extracting regge params
    regge = jnp.array([R[0,1], R[1,0], R[2,2], R[1,1], R[0,0]])
    
    # variable stores '0' is unsuccessful
    did_it_work = jax.lax.cond(issorted(regge),
                               lambda __: 1, 
                               lambda __: 0,
                               operand=None)
    print(regge)

    '''
    # initializing the Regge free parameters from the matrix R
    L, X, T, B, S = regge
    
    #RY03 Eqn.(2.13)
    
    c=L*(24+L*(50+L*(35+L*(10+L))))/120 + \
       X*(6+X*(11+X*(6+X)))/24+T*(2+T*(3+T))/6 + \
       B*(B+1)/2+S+1
    
    return c
    '''
    return regge


if __name__ == "__main__":
    # wigner parameters
    ell1, ell2, ell3 = 2, 10, 12
    m1, m2, m3 = -1, 0, 1 

    # timing the functions with and without jitting
    Niter = 1

    print(find_c_RY03(ell1, ell2, ell3, m1, m2, m3))

    # timing the unjitted version
    t1 = time.time()
    for __ in range(Niter): c = find_c_RY03(ell1, ell2, ell3, m1, m2, m3).block_until_ready()
    t2 = time.time()

    # timing the jitted version
    _find_c_RY03 = jax.jit(find_c_RY03)
    __ = _find_c_RY03(ell1, ell2, ell3, m1, m2, m3)

    t3 = time.time()
    for __ in range(Niter): c = _find_c_RY03(ell1, ell2, ell3, m1, m2, m3).block_until_ready()
    t4 = time.time()


    print('JIT speeds up by: ', (t2-t1)/(t4-t3))
