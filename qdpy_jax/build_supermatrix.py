class build_supermatrix_functions(self):                                                                                                                                                
        """Function that returns the function to calculate                                                                                                                               
        the superMatrix prior to solving eigenvalue problem.                                                                                                                             
        This function is spefic to the central multiplet. So,                                                                                                                            
        CENMULT needs to be a static argument.                                                                                                                                           
        """                           
        def __init__(self):
            pass
                                                                                                                                                   
        def build_supermatrix(self, CENMULT_AND_NBS):                                                                                                                   
            """Function to assimilate all the neighbour info                                                                                                                             
            and return the function to compute the SuperMatrix'                                                                                                                          
            """                                                                                                                                               

            # building the submatrix dictionary
            SUBMAT_DICT = build_SUBMAT_DICT(CENMULT_AND_NBS)

            # tiling supermatrix with submatrices                                                                                                                                        
            supmat = self.tile_submatrices(CENMULT, SUBMAT_DICT)                                                                                                                         

            return supmat                                                                                                                                                                
        
        def build_SUBMAT_DICT(self, CEN_N_NBS):
            """Returns the namedtuple containing the tiling information
            of the submatrices inside the supermatrix.
            """

            dim_blocks = np.size(CEN_N_NBS.omega_nbs)
            
            # supermatix can be tiled with submatrices corresponding to                                                                                                                              
            # (l, n) - (l', n') coupling. The dimensions of the submatrix                                                                                                                            
            # is (2l+1, 2l'+1)                                                                                                                                                                       
            dimX_submat = 2*CEN_N_NBS.nl_nbs[:, 1].reshape(1, dim_blocks) \                                                                                                                             
                          * np.ones((dim_blocks, 1), dtype='int32') + 1                                                                                                                              
            dimY_submat = 2*CEN_N_NBS.nl_nbs[:, 1].reshape(dim_blocks, 1) \                                                                                                                             
                          * np.ones((1, dim_blocks), dtype='int32') + 1    

            # creating the startx, startx, endx, endy for subnatrices                                                                                                                                
            submat_tile_ind = np.zeros((dim_blocks, dim_blocks, 4), dtype='int32')                                                                                                                   

            for ix in range(dim_blocks):                                                                                                                                                             
                for iy in range(dim_blocks):                                                                                                                                                         
                    submat_tile_ind[ix,iy,0] = int(dimX_submat[0, :ix].sum())                                                                                                                        
                    submat_tile_ind[ix,iy,1] = int(dimY_submat[:iy, 0].sum())                                                                                                                        
                    submat_tile_ind[ix,iy,2] = int(dimX_submat[0, :int(ix+1)].sum())                                                                                                                 
                    submat_tile_ind[ix,iy,3] = int(dimY_submat[:int(iy+1), 0].sum()) 
            
            # defining the namedtuple
            SUBMAT_DICT_ = namedtuple('SUBMAT_DICT', ['startx',
                                                      'starty',
                                                      'endx',
                                                      'endy']) 

            SUBMAT_DICT = SUBMAT_DICT_(submat_tile_ind[:,:,0],
                                       submat_tile_ind[:,:,1],
                                       submat_tile_ind[:,:,2],
                                       submat_tile_ind[:,:,3]) 

            return SUBMAT_DICT

        def tile_submatrices(self, CENMULT, SM):                                                                                                                                             
            """Function to loop over the submatrix blocks and tile in the                                                                                                                    
            submatrices into the supermatrix.                                                                                                                                                
            """                                                                                                                                                                              

            supmat = jnp.zeros((CENMULT.dim_super, CENMULT.dim_super), dtype='float32')                                                                                                      
            
            for i in range(CENMULT.dim_blocks):                                                                                                                                              
                for ii in range(i, CENMULT.dim_blocks):                                                                                                                                      
                    startx, starty = SM.startx[i,ii], SM.starty[i,ii]                                                                                                                        
                    endx, endy = SM.endx[i,ii], SM.endy[i,ii]                                                                                                                                
                    
                    # creating the submatrix                                                                                                                                                 
                    submat = jnp.ones((endx-startx, endy-starty), dtype='float32')                                                                                                           
                    
                    supmat = jax.ops.index_update(supmat,                                                                                                                                    
                                                  jax.ops.index[startx:endx, starty:endy],                                                                                                   
                                                  submat)                                                                                                                                    
                    # to avoid repeated filling of the central blocks                                                                                                                        
                    supmat = jax.lax.cond(abs(i-ii)>0,                                                                                                                                       
                                          lambda __: jax.ops.index_update(supmat,                                                                                                            
                                                                          jax.ops.index[starty:endy, startx:endx],                                                                                      
                                                                          jnp.transpose(jnp.conjugate(submat))),                                                                                        
                                          lambda __: supmat,                                                                                                                                 
                                          operand=None)                                                                                                                                      
                    
            return supmat      
