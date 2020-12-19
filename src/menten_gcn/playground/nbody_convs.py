import tensorflow as tf
from tensorflow.keras.layers import *
#from spektral.layers import *

from menten_gcn.util import *


def make_NENE(X, E):
    assert len(X.shape) == 3
    assert len(E.shape) == 4

    Xi_shape = [X.shape[1], 1, X.shape[2]]
    Xj_shape = [1, X.shape[1], X.shape[2]]

    Xi = Reshape(Xi_shape, input_shape=X.shape)(X)
    Xj = Reshape(Xj_shape, input_shape=X.shape)(X)

    Xi = tf.keras.backend.repeat_elements(Xi, rep=X.shape[1], axis=2)
    Xj = tf.keras.backend.repeat_elements(Xj, rep=X.shape[1], axis=1)

    # C1: shape=(None,N,N,S+2F)
    # C2: shape=(None,N,N,nfeatures)

    '''
    if E ==
    1 2 3
    4 5 6
    7 8 9
    then Eprime ==
    1 4 7
    2 5 8
    3 6 9
    '''

    Eprime = tf.transpose(E, perm=[0, 2, 1, 3])
    return Concatenate(axis=-1)([Xi, E, Xj, Eprime])


def expand_E(E):
    #E.shape: (None, N, N, S)
    N = E.shape[1]
    assert(N == E.shape[2])
    S = E.shape[3]
    Ei_shape = [1, N, N, S]
    Ej_shape = [N, 1, N, S]
    Ek_shape = [N, N, 1, S]

    Ei = Reshape(Ei_shape)(E)
    Ej = Reshape(Ej_shape)(E)
    Ek = Reshape(Ek_shape)(E)

    Ei = tf.keras.backend.repeat_elements(Ei, rep=N, axis=1)
    Ej = tf.keras.backend.repeat_elements(Ej, rep=N, axis=2)
    Ek = tf.keras.backend.repeat_elements(Ek, rep=N, axis=3)

    return Ei, Ej, Ek


def make_NEENEENEE(X, E):
    assert len(X.shape) == 3
    assert len(E.shape) == 4

    N = X.shape[1]
    F = X.shape[2]
    S = E.shape[3]

    Xi_shape = [N, 1, 1, F]
    Xj_shape = [1, N, 1, F]
    Xk_shape = [1, 1, N, F]

    Xi = Reshape(Xi_shape)(X)
    Xj = Reshape(Xj_shape)(X)
    Xk = Reshape(Xk_shape)(X)

    Xi = tf.keras.backend.repeat_elements(Xi, rep=N, axis=2)
    Xi = tf.keras.backend.repeat_elements(Xi, rep=N, axis=3)
    Xj = tf.keras.backend.repeat_elements(Xj, rep=N, axis=1)
    Xj = tf.keras.backend.repeat_elements(Xj, rep=N, axis=3)
    Xk = tf.keras.backend.repeat_elements(Xk, rep=N, axis=1)
    Xk = tf.keras.backend.repeat_elements(Xk, rep=N, axis=2)

    #print( Xi.shape, Xj.shape, Xk.shape )
    Ei, Ej, Ek = expand_E(E)
    Eprime = tf.transpose(E, perm=[0, 2, 1, 3])
    Eti, Etj, Etk = expand_E(Eprime)

    C = Concatenate(axis=-1)([Xi, Ei, Eti, Xj, Ej, Etj, Xk, Ek, Etk])
    #print( C.shape )
    target_shape = (None, N, N, N, (3 * F) + (6 * S))
    #print( target_shape )
    for i in range(1, 5):
        assert(C.shape[i] == target_shape[i])
    return C, Eprime


def make_NEENEENEE_mask(E_mask):
    assert len(E_mask.shape) == 4
    Ei, Ej, Ek = expand_E(E_mask)
    return Multiply()([Ei, Ej, Ek])


def make_NENE_XE_conv(X, A, E, Xnfeatures: list, Enfeatures: int, Xactivation='relu', Eactivation='relu', E_mask=None, X_mask=None):
    """
    We find that current GCN layers undervalue the Edge tensors.
    Not only does this layer use them as input,
    it also updates the values of Edge tensors.

    Disclaimer: this isn't actually a layer at the moment.
    It's a method that hacks layers together and returns the result.

    Parameters
    ---------
    X: layer
        Node features
    A: layer
        Adjaceny matrix
    E: layer
        Edge features
    Xnfeatures: list of 2 ints
        How many features do you want each node to end up with?
        At the moment this asks you to give two values because there is an intermediate layer.
        Xnfeatures=[64,32] will end up with 32 values for each Node.
    Enfeatures: int
        How many features do you want each edge to end up with?
    Xactivation:
        Which activation function should be applied to the final X?
    Eactivation:
        Which activation function should be applied to the final E?
    E_mask: layer
        If you already made an edge mask, feel free to pass it here to save us time.
    X_mask: layer
        If you already made a node mask, feel free to pass it here to save us time.

    Returns
    ---------
    - keras layer which is the new X
    - keras layer which is the new E
    """

    # X: shape=(None,N,F)
    # A: shape=(None,N,N)
    # E: shape=(None,N,N,S)

    assert len(X.shape) == 3
    assert len(A.shape) == 3
    assert len(E.shape) == 4

    NENE = make_NENE(X, E)

    newE = Conv2D(filters=Enfeatures, kernel_size=1, activation=Eactivation)(NENE)
    if E_mask is None:
        E_mask = make_edge_mask(A)
    newE = apply_edge_mask(E=newE, E_mask=E_mask)

    newX = Conv2D(filters=Xnfeatures[0], kernel_size=1, activation=None)(NENE)
    newX = apply_edge_mask(E=newX, E_mask=E_mask)
    newX1 = tf.keras.backend.sum(newX, axis=-2, keepdims=False)
    newX1 = PReLU(shared_axes=[1])(newX1)
    newX2 = tf.keras.backend.sum(newX, axis=-3, keepdims=False)
    newX2 = PReLU(shared_axes=[1])(newX2)
    newX = Concatenate(axis=-1)([X, newX1, newX2])

    newX = Conv1D(filters=Xnfeatures[1], kernel_size=1, activation=Xactivation)(newX)

    if X_mask is None:
        X_mask = make_node_mask(A)
    newX = apply_node_mask(X=newX, X_mask=X_mask)

    return newX, newE


def make_NEENEENEE_XE_conv(X, A, E, Tnfeatures: list, Xnfeatures: int, Enfeatures: int,
                           Xactivation='relu', Eactivation='relu', E_mask=None, X_mask=None):
    """
    Same idea as make_NENE_XE_conv but considers all possible 3-body interactions. Warning: this will use a ton of memory if your graph is large.

    Disclaimer: this isn't actually a layer at the moment.
    It's a method that hacks layers together and returns the result.

    Parameters
    ---------
    X: layer
        Node features
    A: layer
        Adjaceny matrix
    E: layer
        Edge features
    Tnfeatures: list of ints
        This time, you get to decide the number of middle layers.
        Make this list as long as you want
    Xnfeatures: int
        How many features do you want each node to end up with?
    Enfeatures: int
        How many features do you want each edge to end up with?
    Xactivation:
        Which activation function should be applied to the final X?
    Eactivation:
        Which activation function should be applied to the final E?
    E_mask: layer
        If you already made an edge mask, feel free to pass it here to save us time.
    X_mask: layer
        If you already made a node mask, feel free to pass it here to save us time.

    Returns
    ---------
    - keras layer which is the new X
    - keras layer which is the new E
    """

    # X: shape=(None,N,F)
    # A: shape=(None,N,N)
    # E: shape=(None,N,N,S)

    assert len(X.shape) == 3
    assert len(A.shape) == 3
    assert len(E.shape) == 4

    if E_mask is None:
        E_mask = make_edge_mask(A)

    if X_mask is None:
        X_mask = make_node_mask(A)

    NEE3, Et = make_NEENEENEE(X, E)

    if hasattr(Tnfeatures, "__len__"):
        Temp = NEE3
        for t in Tnfeatures:
            Temp = Conv3D(filters=t, kernel_size=1, activation=None)(Temp)
    else:
        Temp = Conv3D(filters=Tnfeatures, kernel_size=1, activation=None)(NEE3)

    mask = make_NEENEENEE_mask(E_mask)
    Temp = Multiply()([Temp, mask])

    Xi = tf.keras.backend.sum(Temp, axis=[-4, -3], keepdims=False)
    #print( Xi.shape )
    Xj = tf.keras.backend.sum(Temp, axis=[-4, -2], keepdims=False)
    Xk = tf.keras.backend.sum(Temp, axis=[-3, -2], keepdims=False)
    X = Concatenate(axis=-1)([X, Xi, Xj, Xk])  # Activation here?
    newX = Conv1D(filters=Xnfeatures, kernel_size=1, activation=Xactivation)(X)

    Ei = tf.keras.backend.sum(Temp, axis=[-4], keepdims=False)
    #print( Ei.shape )
    Ek = tf.keras.backend.sum(Temp, axis=[-3], keepdims=False)
    Ej = tf.keras.backend.sum(Temp, axis=[-2], keepdims=False)
    Eti = tf.transpose(Ei, perm=[0, 2, 1, 3])
    Etj = tf.transpose(Ej, perm=[0, 2, 1, 3])
    Etk = tf.transpose(Ek, perm=[0, 2, 1, 3])
    E = Concatenate(axis=-1)([E, Et, Ei, Eti, Ej, Etj, Ek, Etk])
    #print( E.shape )
    newE = Conv2D(filters=Enfeatures, kernel_size=1, activation=Eactivation)(E)

    newE = apply_edge_mask(E=newE, E_mask=E_mask)
    newX = apply_node_mask(X=newX, X_mask=X_mask)
    return newX, newE


def add_n_edges_for_node(X, A):
    #print( A.shape )
    n_edges = tf.keras.backend.mean(A, axis=-1, keepdims=False)
    #print( n_edges.shape )
    n_edges = Reshape((X.shape[1], 1))(n_edges)
    #print( n_edges.shape )
    newX = Concatenate(axis=-1)([X, n_edges])
    return newX
