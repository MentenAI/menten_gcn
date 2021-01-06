from tensorflow.keras.layers import MaxPooling1D, Reshape, Multiply, Layer
from menten_gcn.wrappers import WrappedPose
import numpy as np

##############
# NODE MASKS #
##############


def make_node_mask(A: Layer) -> Layer:
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


def apply_node_mask(X: Layer, X_mask: Layer) -> Layer:
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


def make_and_apply_node_mask(X: Layer, A: Layer) -> Layer:
    """
    Sometimes you want to zero-out rows of your X layer that are not currently populated with a node.
    This method applies a mask from the Adjaceny matrix to do that.

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


def make_edge_mask(A: Layer) -> Layer:
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


def apply_edge_mask(E: Layer, E_mask: Layer) -> Layer:
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


def make_and_apply_edge_mask(E: Layer, A: Layer) -> Layer:
    """
    Sometimes you want to zero-out elements of your E layer that are not currently populated with an edge.
    This method applies a mask from the Adjaceny matrix to do that.

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


##############
# CLUSTERING #
##############

'''
USING CA:
max_dist_A=20.0 seems to result in a max of 45 residues being selected
max_dist_A=15.0 seems to result in a max of 25 residues being selected
max_dist_A=12.0 seems to result in a max of 15 residues being selected
max_dist_A=10.0 seems to result in a max of 10 residues being selected
'''


def make_adjacency_matrix_for_clustering(wrapped_pose: WrappedPose, focus_resids: list, max_dist_A: float, CB: bool = False):
    n_foc = len(focus_resids)
    assert n_foc > 0
    A = np.zeros(shape=(n_foc, n_foc))
    for i in range(0, n_foc - 1):
        resid_i = focus_resids[i]
        if CB:
            xyz_i = wrapped_pose.approximate_ALA_CB(resid_i)
        else:
            xyz_i = wrapped_pose.get_atom_xyz(resid_i, "CA")
        for j in range(i + 1, n_foc):
            resid_j = focus_resids[j]
            if CB:
                xyz_j = wrapped_pose.approximate_ALA_CB(resid_j)
            else:
                xyz_j = wrapped_pose.get_atom_xyz(resid_j, "CA")
            dist = np.linalg.norm(xyz_i - xyz_j)
            if dist <= max_dist_A:
                A[i][j] = 1
                A[j][i] = 1
    return A


def cluster(wrapped_pose: WrappedPose, focus_resids: list, max_dist_A: float = 20.0, CB: bool = False) -> list:
    A = make_adjacency_matrix_for_clustering(wrapped_pose, focus_resids, max_dist_A, CB=CB)
    clusters = []
    n_foc = len(focus_resids)
    n_unassigned = n_foc

    is_assigned = [False for _ in range(0, n_foc)]

    while n_unassigned > 0:
        min_i = -1
        min_count = 99999
        for i in range(0, n_foc):
            if is_assigned[i]:
                continue
            sum_i = np.sum(A[i])
            if sum_i < min_count:
                min_count = sum_i
                min_i = i
        assert min_i >= 0

        cluster = []
        cluster.append(focus_resids[min_i])
        is_assigned[min_i] = True
        for j in range(0, n_foc):
            if A[min_i][j] == 1:
                cluster.append(focus_resids[j])
                is_assigned[j] = True
                A[j, :] = 0
                A[:, j] = 0
        A[min_i, :] = 0  # not needed?
        A[:, min_i] = 0  # not needed?

        n_unassigned -= len(cluster)
        clusters.append(cluster)

    return clusters


def cluster_all_resids(wrapped_pose: WrappedPose, max_dist_A: float = 20.0, CB: bool = False) -> list:
    return cluster(wrapped_pose, focus_resids=[i for i in range(1, wrapped_pose.n_residues() + 1)], max_dist_A=max_dist_A, CB=CB)
