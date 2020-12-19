from tensorflow.keras.layers import *

##############
# NODE MASKS #
##############


def make_node_mask(A):
    """
    Create a node mask here, then re-use it many times in the future

    Parameters
    ---------
    A: layer
        Adjaceny matrix

    Returns
    ---------
    - keras layer
    """
    pool_size = A.shape[2]
    X_mask1 = MaxPooling1D(pool_size=pool_size)(A)
    X_mask_shape = (X_mask1.shape + (1,))[2:]
    X_mask = Reshape(X_mask_shape)(X_mask1)
    return X_mask


def apply_node_mask(X, X_mask):
    """
    Apply a mask that you've already made

    Parameters
    ---------
    X: layer
        Node features
    X_mask: layer
        Mask created by make_node_mask

    Returns
    ---------
    - keras layer
    """
    return Multiply()([X, X_mask])


def make_and_apply_node_mask(X, A):
    """
    Sometimes you want to zero-out rows of your X layer that are not currently populated with a node. This method applies a mask from the Adjaceny matrix to do that.

    Parameters
    ---------
    X: layer
        Node features
    A: layer
        Adjaceny matrix

    Returns
    ---------
    - keras layer
    """
    return apply_node_mask(X, make_node_mask(A))

##############
# EDGE MASKS #
##############


def make_edge_mask(A):
    """
    Create an edge mask here, then re-use it many times in the future

    Parameters
    ---------
    A: layer
        Adjaceny matrix

    Returns
    ---------
    - keras layer
    """

    E_mask_shape = (A.shape[1], A.shape[2], 1)
    return Reshape(E_mask_shape, input_shape=A.shape)(A)


def apply_edge_mask(E, E_mask):
    """
    Apply a mask that you've already made

    Parameters
    ---------
    E: layer
        Edge features
    E_mask: layer
        Mask created by make_edge_mask

    Returns
    ---------
    - keras layer
    """
    return Multiply()([E, E_mask])


def make_and_apply_edge_mask(E, A):
    """
    Sometimes you want to zero-out elements of your E layer that are not currently populated with an edge. This method applies a mask from the Adjaceny matrix to do that.

    Parameters
    ---------
    E: layer
        Edge features
    A: layer
        Adjaceny matrix

    Returns
    ---------
    - keras layer
    """

    assert len(E.shape) == len(A.shape) + 1
    return apply_edge_mask(E, make_edge_mask(A))
