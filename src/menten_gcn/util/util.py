from tensorflow.keras.layers import *

##############
# NODE MASKS #
##############

def make_node_mask( A ):
    pool_size = A.shape[2]
    X_mask1 = MaxPooling1D( pool_size=pool_size )( A )
    X_mask_shape = (X_mask1.shape + (1,))[2:]
    #print( X_mask_shape )
    X_mask = Reshape( X_mask_shape )( X_mask1 )
    #print( X_mask.shape )
    #exit( 0 )    
    return X_mask
    

def apply_node_mask( X, X_mask ):
    #TODO assert shapes
    return Multiply()([ X, X_mask ])

def make_and_apply_node_mask( X, A ):
    return apply_node_mask( X, make_node_mask( A ) )

##############
# EDGE MASKS #
##############

def make_edge_mask( A ):
    #TODO assert shapes
    E_mask_shape = ( A.shape[1], A.shape[2], 1 )
    return Reshape( E_mask_shape, input_shape=A.shape )( A )

def apply_edge_mask( E, E_mask ):
    #TODO assert shapes    
    return Multiply()([ E, E_mask ])
    
def make_and_apply_edge_mask( E, A ):
    assert len( E.shape ) == len( A.shape ) + 1
    return apply_edge_mask( E, make_edge_mask( A ) )
