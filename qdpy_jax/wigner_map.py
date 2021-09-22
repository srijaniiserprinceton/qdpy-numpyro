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
    
    temp = jnp.zeros((1,2), dtype='int32')
    
    # finding the location of the smallest elements S
    S_cont_index = jnp.argmin(R)
    S_row_ind, S_col_ind = ind2sub(S_cont_index, 3, 3)
    temp = jax.ops.index_update(temp, 0, S_row_ind)
    temp = jax.ops.index_update(temp, 1, S_col_ind)
    
    R = circshift(R,-(temp-1));
    
    # finding the location of the largest element L
    L_cont_ind = jnp.argmax(R)
    L_row_ind, L_col_ind = ind2sub(L_cont_ind, 3, 3)
    temp = jax.ops.index_update(temp, 0, L_row_ind)
    temp = jax.ops.index_update(temp, 1, L_col_ind)
    
    #Reorder Regge square
    jax.lax.cond(
        temp[0] == 1,
        R = jax.numpy.transpose(R),
        lambda __: None,
        operand=none)
        if(temp(1)==3):
            R(:,2:3) = jnp.fliplr(R(:,2:3));
            oddperm(i)=true;

    else:
        if temp(2)==3
        R(:,2:3)=fliplr(R(:,2:3));
        oddperm(i)=true;
    

    if R(3,2)<R(2,2):
        oddperm(i)=1-oddperm(i);
        R(2:3,:)=flipud(R(2:3,:));
    elif R(3,2)==R(2,2) && R(3,3)<R(2,3);
R(2:3,:)=flipud(R(2:3,:));
oddperm(i)=1-oddperm(i);
end
regge(i,1:5)=[R(1,2) R(2,1) R(3,3) R(2,2) R(1,1)]; 
if ~issorted(wrev(regge(i,:)))
disp('WHOA! something''s wrong here...')
end
L, X, T, B, S = regge

#RY03 Eqn.(2.13)

c=L*(24+L*(50+L*(35+L*(10+L))))/120 + \
      X*(6+X*(11+X*(6+X)))/24+T*(2+T*(3+T))/6 + \
      B*(B+1)/2+S+1
