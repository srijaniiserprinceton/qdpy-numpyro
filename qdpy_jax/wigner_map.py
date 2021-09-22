import jax.numpy as jnp
import jax

# function to find 2-d index for a contiguous numbering
# of elements in a matrix
def ind2sub(cont_ind, nrows, ncols):
    return cont_ind//nrows, cont_ind%ncols

def find_c_RY03():
    # following Rasch and Yu, 2003 (RY03)
    # initializing a sample Wigner-3j
    # /ell1 ell2 ell3\
    # \ m1   m2   m3 /
    ell1, ell2, ell3 = 2, 10, 12 
    m1, m2, m3 = -1, 0, 1
    
    # forming the R matrix according to Eqn.(2.10) in RY03
    R = jnp.array([[-ell1 + ell2 + ell3, ell1 - ell2 + ell3, ell1 + ell2 - ell3],
                   [ell1 - m1, ell2 - m2, ell3 - m3],
                   [ell1 + m1, ell2 + m2, ell3 + m3]]) 
    
    # converting it to the Eqn.(2.11) form according to the step in
    # ~https://github.com/csdms-contrib/slepian_alpha/blob/master/wignersort.m~

    # to store various indices during the operation
    temp = jnp.zeros((1,2), dtype='int32')
    # to store information of the phase re-adjustment needed
    oddperm = np.array([0], dtype='boolean')

    # finding the location of the smallest elements S
    S_cont_index = jnp.argmin(R)
    S_row_ind, S_col_ind = ind2sub(S_cont_index, 3, 3)
    temp = jax.ops.index_update(temp, 0, S_row_ind)
    temp = jax.ops.index_update(temp, 1, S_col_ind)
    
    # moving S to the (0,0) position
    R = jnp.roll(R, -(temp[0]-1), axis=0)
    R = jnp.roll(R, -(temp[1]-1), axis=1)

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

    def true_func_1():
        R = jnp.transpose(R)
        def true_func_1_true_func():
            R = jax.ops.index_update(R,jax.ops.index[:,1:3],jnp.fliplr(b[:,1:3])))
            oddperm = jax.ops.index_add(oddperm,0,1)
            return R, oddperm
            
        R, oddperm = jax.ops.cond(temp[0] == 2,
                    true_func_1_true_func,
                    lambda __: return R, oddperm,
                    operand=True)
        
        return R, oddperm

    def false_func_1():
        def false_func_1_true_func():
            R = jax.ops.index_update(R,jax.ops.index[:,1:3],jnp.fliplr(b[:,1:3])))
            oddperm = jax.ops.index_add(oddperm,0,1)
            return R, oddperm

    R, oddperm = jax.ops.cond(temp[1] == 0,
                              true_func_1,
                              false_func_1,
                              operand=None)


    def true_func_2():
        oddperm = jax.ops.index_update(oddperm, 0, 1-oddperm[0])
        R = jax.ops.index_update(R,jax.ops.index[1:3,:],jnp.flipud(b[1:3,:])))
        return R, oddperm

    def false_func_2():
        def flase_func_2_true_func():
            R = jax.ops.index_update(R,jax.ops.index[1:3,:],jnp.flipud(b[1:3,:])))
            oddperm = jax.ops.index_update(oddperm, 0, 1-oddperm[0])
            return R, oddperm

        jax.ops.cond(R(3,2) == R(2,2) and R(3,3) < R(2,3),
                     false_func_2_true_func,
                     lambda __: return None,
                     operand=None)

    jax.ops.cond(R(3,2) < R(2,2),
        true_func_2,
        false_func_2,
        operand=None)
    
    
    if ~issorted(wrev(regge(i,:)))
    disp('WHOA! something''s wrong here...')
    end

    # initializing the Regge free parameters from the matrix R
    L, X, T, B, S = regge
    
    #RY03 Eqn.(2.13)
    
    c=L*(24+L*(50+L*(35+L*(10+L))))/120 + \
       X*(6+X*(11+X*(6+X)))/24+T*(2+T*(3+T))/6 + \
       B*(B+1)/2+S+1
