import tensorflow as tf
from tensorflow.keras.layers import *
#from spektral.layers import *

from menten_gcn.util import *

from typing import Tuple


def make_NENE(X: Layer, E: Layer) -> Layer:
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


def expand_E(E: Layer) -> Tuple[Layer, Layer, Layer]:
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


def make_NEENEENEE(X: Layer, E: Layer) -> Tuple[Layer, Layer]:
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


def make_NEENEENEE_mask(E_mask: Layer) -> Layer:
    assert len(E_mask.shape) == 4
    Ei, Ej, Ek = expand_E(E_mask)
    return Multiply()([Ei, Ej, Ek])


def make_1body_conv(X: Layer, A: Layer, E: Layer,
                    Xnfeatures: int, Enfeatures: int,
                    Xactivation='relu', Eactivation='relu',
                    E_mask=None, X_mask=None) -> Tuple[Layer, Layer]:
    newX = Conv1D(filters=Xnfeatures, kernel_size=1, activation=Xactivation)(X)
    if X_mask is None:
        X_mask = make_node_mask(A)
    newX = apply_node_mask(X=newX, X_mask=X_mask)

    newE = Conv2D(filters=Enfeatures, kernel_size=1, activation=Eactivation)(E)
    if E_mask is None:
        E_mask = make_edge_mask(A)
    newE = apply_edge_mask(E=newE, E_mask=E_mask)

    return newX, newE


def make_NENE_XE_conv(X: Layer, A: Layer, E: Layer,
                      Tnfeatures: list, Xnfeatures: int, Enfeatures: int,
                      Xactivation='relu', Eactivation='relu',
                      attention: bool = False, apply_T_to_E: bool = False,
                      E_mask=None, X_mask=None) -> Tuple[Layer, Layer]:
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
    Tnfeatures: list of ints
        How large should each intermediate layer be?
        The length of this list determines the number of intermediate layers.
    Xnfeatures: int
        How many features do you want each node to end up with?
    Enfeatures: int
        How many features do you want each edge to end up with?
    Xactivation:
        Which activation function should be applied to the final X?
    Eactivation:
        Which activation function should be applied to the final E?
    attention: bool
        Should we apply attention weights to the sum operations?
    apply_T_to_E: bool
        Should the input to the final E conv be the Temp tensor or the initial NENE?
        Feel free to just use the default if that question makes no sense
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

    if X_mask is None:
        X_mask = make_node_mask(A)
    if E_mask is None:
        E_mask = make_edge_mask(A)

    NENE = make_NENE(X, E)
    Temp = NENE

    if hasattr(Tnfeatures, "__len__"):
        assert len(Tnfeatures) > 0
        for t in Tnfeatures:
            Temp = Conv2D(filters=t, kernel_size=1, activation=PReLU(shared_axes=[1, 2]))(Temp)
    else:
        Temp = Conv2D(filters=Tnfeatures, kernel_size=1, activation=PReLU(shared_axes=[1, 2]))(Temp)

    Temp = apply_edge_mask(E=Temp, E_mask=E_mask)

    if attention:
        Att1 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att1 = Multiply()([Temp, Att1])
        newX1 = tf.keras.backend.sum(Att1, axis=-2, keepdims=False)

        Att2 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att2 = Multiply()([Temp, Att2])
        newX2 = tf.keras.backend.sum(Att2, axis=-3, keepdims=False)
    else:
        newX1 = tf.keras.backend.sum(Temp, axis=-2, keepdims=False)
        newX2 = tf.keras.backend.sum(Temp, axis=-3, keepdims=False)

    #newX1 = PReLU(shared_axes=[1])(newX1)
    #newX2 = PReLU(shared_axes=[1])(newX2)
    superX = Concatenate(axis=-1)([X, newX1, newX2])

    if apply_T_to_E:
        superE = Temp
    else:
        superE = NENE

    newX, newE = make_1body_conv(superX, A, superE, Xnfeatures, Enfeatures,
                                 Xactivation, Eactivation, E_mask, X_mask)

    return newX, newE




def make_flat_NENE1(X, A, E):
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
    NENE = Concatenate(axis=-1)([Xi, E, Xj, Eprime])
    
    NENE = make_NENE(X, E)  # TODO make_flat_NENE
    A_int = tf.cast(A, "int32")
    part = tf.dynamic_partition(NENE, A_int, 2)
    flat_NENE = part[1]
    return NENE, A_int, flat_NENE

def make_flat_NENE2(X, A, E):
    assert len(X.shape) == 3
    assert len(E.shape) == 4

    A_int = tf.cast(A, "int32")
    
    Xi_shape = [X.shape[1], 1, X.shape[2]]
    Xj_shape = [1, X.shape[1], X.shape[2]]

    Xi = Reshape(Xi_shape, input_shape=X.shape)(X)
    Xj = Reshape(Xj_shape, input_shape=X.shape)(X)

    Xi = tf.keras.backend.repeat_elements(Xi, rep=X.shape[1], axis=2)
    Xj = tf.keras.backend.repeat_elements(Xj, rep=X.shape[1], axis=1)

    Xi_flat = tf.dynamic_partition(Xi, A_int, 2)[1]
    Xj_flat = tf.dynamic_partition(Xj, A_int, 2)[1]
    
    Et = tf.transpose(E, perm=[0, 2, 1, 3])
    E_flat = tf.dynamic_partition(E, A_int, 2)[1]
    Et_flat = tf.dynamic_partition(Et, A_int, 2)[1]

    flat_NENE = Concatenate(axis=-1)([Xi_flat, E_flat, Xj_flat, Et_flat])
    return A_int, flat_NENE


def flat2_unnamed_util(n, A_int, final_t):
    #x = tf.constant(N*N)
    r = tf.range(n * n * tf.shape(A_int)[0])
    r2 = tf.reshape(r, shape=[tf.shape(A_int)[0], n, n])
    condition_indices = tf.dynamic_partition(r2, A_int, 2)

    s_1 = tf.shape(condition_indices[0])[0]
    s_2 = final_t
    s = [s_1, s_2]
    zero_padding1 = tf.zeros(shape=s)
    return condition_indices, zero_padding1


def flat2_unnamed_util2(A_int, n, final_t, Temp):
    #       batch              * N * N * t
    npad1 = tf.shape(A_int)[0] * n * n * final_t
    npad2 = tf.shape(Temp)[0]
    nz = npad1 - npad2
    zero_padding = tf.zeros(nz, dtype=Temp.dtype)
    zero_padding = tf.reshape(zero_padding, [-1, final_t])
    return zero_padding


def flat2_deflatten(V, condition_indices, zero_padding1, A_int, final_t, n):
    partitioned_data = [zero_padding1, V]
    V = tf.dynamic_stitch(condition_indices, partitioned_data)

    zero_padding = flat2_unnamed_util2(A_int, n, final_t, V)

    V = tf.concat([V, zero_padding], -2)
    V = tf.reshape(V, [tf.shape(A_int)[0], n, n, final_t])
    return V


def make_flat_2body_conv(X: Layer, A: Layer, E: Layer,
                         Tnfeatures: list, Xnfeatures: int, Enfeatures: int,
                         Xactivation='relu', Eactivation='relu',
                         attention: bool = False, apply_T_to_E: bool = False,
                         E_mask=None, X_mask=None) -> Tuple[Layer, Layer]:
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
    Tnfeatures: list of ints
        How large should each intermediate layer be?
        The length of this list determines the number of intermediate layers.
    Xnfeatures: int
        How many features do you want each node to end up with?
    Enfeatures: int
        How many features do you want each edge to end up with?
    Xactivation:
        Which activation function should be applied to the final X?
    Eactivation:
        Which activation function should be applied to the final E?
    attention: bool
        Should we apply attention weights to the sum operations?
    apply_T_to_E: bool
        Should the input to the final E conv be the Temp tensor or the initial NENE?
        Feel free to just use the default if that question makes no sense
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

    N = A.shape[-1]

    if X_mask is None:
        X_mask = make_node_mask(A)
    if E_mask is None:
        E_mask = make_edge_mask(A)

    A_int, flat_NENE = make_flat_NENE2(X, A, E)
    Temp = flat_NENE
    
    if hasattr(Tnfeatures, "__len__"):
        assert len(Tnfeatures) > 0
        for t in Tnfeatures:
            Temp = Dense(t, activation=PReLU())(Temp)
            final_t = t
    else:
        Temp = Dense(Tnfeatures, activation=PReLU())(Temp)
        final_t = Tnfeatures

    n = tf.constant(N)
    condition_indices, zero_padding1 = flat2_unnamed_util(n, A_int, final_t)

    Temp_final_flat = Temp
    
    if attention:
        Att1 = Dense(1, activation='sigmoid')(Temp)
        Att1 = Multiply()([Temp, Att1])
        Att1 = flat2_deflatten(Att1, condition_indices,
                               zero_padding1, A_int, final_t, n)
        newX1 = tf.keras.backend.sum(Att1, axis=-2, keepdims=False)

        Att2 = Dense(1, activation='sigmoid')(Temp)
        Att2 = Multiply()([Temp, Att2])
        Att2 = flat2_deflatten(Att2, condition_indices,
                               zero_padding1, A_int, final_t, n)
        newX2 = tf.keras.backend.sum(Att2, axis=-3, keepdims=False)
    else:
        Temp = flat2_deflatten(Temp, condition_indices,
                               zero_padding1, A_int, final_t, n)
        newX1 = tf.keras.backend.sum(Temp, axis=-2, keepdims=False)
        newX2 = tf.keras.backend.sum(Temp, axis=-3, keepdims=False)

    superX = Concatenate(axis=-1)([X, newX1, newX2])

    if apply_T_to_E:
        superE = Temp_final_flat
    else:
        superE = flat_NENE

    newE = Dense(Enfeatures, activation=Eactivation)( superE )
    condition_indices, zero_padding1 = flat2_unnamed_util(n, A_int, Enfeatures)
    newE = flat2_deflatten( newE, condition_indices, zero_padding1,
                            A_int, Enfeatures, n )


    dummy = E #Doesn't matter
    newX, _ = make_1body_conv(superX, A, dummy, Xnfeatures, Enfeatures,
                                 Xactivation, Eactivation, E_mask, X_mask)

    return newX, newE


def make_NEENEENEE_XE_conv(X: Layer, A: Layer, E: Layer,
                           Tnfeatures: list, Xnfeatures: int,
                           Enfeatures: int, Xactivation='relu',
                           Eactivation='relu', attention: bool = False,
                           E_mask=None, X_mask=None) -> Tuple[Layer, Layer]:
    """
    Same idea as make_NENE_XE_conv but considers all possible 3-body interactions.
    Warning: this will use a ton of memory if your graph is large.

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
    attention: bool
        Should we apply attention weights to the sum operations?
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
            Temp = Conv3D(filters=t, kernel_size=1,
                          activation=PReLU(shared_axes=[1, 2, 3]))(Temp)
    else:
        Temp = Conv3D(filters=Tnfeatures, kernel_size=1,
                      activation=PReLU(shared_axes=[1, 2, 3]))(NEE3)

    mask = make_NEENEENEE_mask(E_mask)
    Temp = Multiply()([Temp, mask])

    if attention:
        Att_xi = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att_xj = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att_xk = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att_ei = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att_ej = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)
        Att_ek = Conv3D(filters=1, kernel_size=1, activation='sigmoid')(Temp)

        Att_xi = Multiply()([Temp, Att_xi])
        Att_xj = Multiply()([Temp, Att_xj])
        Att_xk = Multiply()([Temp, Att_xk])
        Att_ei = Multiply()([Temp, Att_ei])
        Att_ej = Multiply()([Temp, Att_ej])
        Att_ek = Multiply()([Temp, Att_ek])

        Xi = tf.keras.backend.sum(Att_xi, axis=[-4, -3], keepdims=False)
        Xj = tf.keras.backend.sum(Att_xj, axis=[-4, -2], keepdims=False)
        Xk = tf.keras.backend.sum(Att_xk, axis=[-3, -2], keepdims=False)

        Ei = tf.keras.backend.sum(Att_ei, axis=[-4], keepdims=False)
        Ek = tf.keras.backend.sum(Att_ej, axis=[-3], keepdims=False)
        Ej = tf.keras.backend.sum(Att_ek, axis=[-2], keepdims=False)
    else:
        Xi = tf.keras.backend.sum(Temp, axis=[-4, -3], keepdims=False)
        Xj = tf.keras.backend.sum(Temp, axis=[-4, -2], keepdims=False)
        Xk = tf.keras.backend.sum(Temp, axis=[-3, -2], keepdims=False)

        Ei = tf.keras.backend.sum(Temp, axis=[-4], keepdims=False)
        Ek = tf.keras.backend.sum(Temp, axis=[-3], keepdims=False)
        Ej = tf.keras.backend.sum(Temp, axis=[-2], keepdims=False)

    superX = Concatenate(axis=-1)([X, Xi, Xj, Xk])  # Activation here?

    Eti = tf.transpose(Ei, perm=[0, 2, 1, 3])
    Etj = tf.transpose(Ej, perm=[0, 2, 1, 3])
    Etk = tf.transpose(Ek, perm=[0, 2, 1, 3])
    superE = Concatenate(axis=-1)([E, Et, Ei, Eti, Ej, Etj, Ek, Etk])

    newX, newE = make_1body_conv(superX, A, superE, Xnfeatures, Enfeatures,
                                 Xactivation, Eactivation, E_mask, X_mask)

    return newX, newE


def add_n_edges_for_node(X: Layer, A: Layer) -> Layer:
    #print( A.shape )
    n_edges = tf.keras.backend.mean(A, axis=-1, keepdims=False)
    #print( n_edges.shape )
    n_edges = Reshape((X.shape[1], 1))(n_edges)
    #print( n_edges.shape )
    newX = Concatenate(axis=-1)([X, n_edges])
    return newX


make_2body_conv = make_NENE_XE_conv

make_3body_conv = make_NEENEENEE_XE_conv
