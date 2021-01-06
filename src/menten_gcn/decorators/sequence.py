import numpy as np

from menten_gcn.decorators.base import Decorator


class Sequence(Decorator):

    """
    One-hot encode the canonical amino acid identity on each node.

    - 20 Node Features
    - 0 Edge Features
    """

    def get_version_name(self):
        return "Sequence"

    def __init__(self):
        pass

    def n_node_features(self):
        return 20

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        name = wrapped_pose.get_name1(resid)
        AAs = "ACDEFGHIKLMNPQRSTVWY"
        onehot = np.zeros(20)
        for i in range(0, 20):
            if name == AAs[i]:
                onehot[i] = 1.0
                break
        return onehot

    def describe_node_features(self):
        return [
            "1 if residue is A, 0 otherwise",
            "1 if residue is C, 0 otherwise",
            "1 if residue is D, 0 otherwise",
            "1 if residue is E, 0 otherwise",
            "1 if residue is F, 0 otherwise",
            "1 if residue is G, 0 otherwise",
            "1 if residue is H, 0 otherwise",
            "1 if residue is I, 0 otherwise",
            "1 if residue is K, 0 otherwise",
            "1 if residue is L, 0 otherwise",
            "1 if residue is M, 0 otherwise",
            "1 if residue is N, 0 otherwise",
            "1 if residue is P, 0 otherwise",
            "1 if residue is Q, 0 otherwise",
            "1 if residue is R, 0 otherwise",
            "1 if residue is S, 0 otherwise",
            "1 if residue is T, 0 otherwise",
            "1 if residue is V, 0 otherwise",
            "1 if residue is W, 0 otherwise",
            "1 if residue is Y, 0 otherwise",
        ]

    def n_edge_features(self):
        return 0


class DesignableSequence(Decorator):

    """
    One-hot encode the canonical amino acid identity on each node,
    with a 21st value for residues that are not yet
    assigned an amino acid identity.

    Note: requires you to call WrappedPose.set_designable_resids first

    - 21 Node Features
    - 0 Edge Features
    """

    def get_version_name(self):
        return "DesignableSequence"

    def n_node_features(self):
        return 21

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        onehot = np.zeros(21)
        if wrapped_pose.resid_is_designable(resid):
            onehot[20] = 1.0
        else:
            name = wrapped_pose.get_name1(resid)
            AAs = "ACDEFGHIKLMNPQRSTVWY"
            for i in range(0, 20):
                if name == AAs[i]:
                    onehot[i] = 1.0
                    break
        return onehot

    def describe_node_features(self):
        return [
            "1 if residue is A and not designable, 0 otherwise",
            "1 if residue is C and not designable, 0 otherwise",
            "1 if residue is D and not designable, 0 otherwise",
            "1 if residue is E and not designable, 0 otherwise",
            "1 if residue is F and not designable, 0 otherwise",
            "1 if residue is G and not designable, 0 otherwise",
            "1 if residue is H and not designable, 0 otherwise",
            "1 if residue is I and not designable, 0 otherwise",
            "1 if residue is K and not designable, 0 otherwise",
            "1 if residue is L and not designable, 0 otherwise",
            "1 if residue is M and not designable, 0 otherwise",
            "1 if residue is N and not designable, 0 otherwise",
            "1 if residue is P and not designable, 0 otherwise",
            "1 if residue is Q and not designable, 0 otherwise",
            "1 if residue is R and not designable, 0 otherwise",
            "1 if residue is S and not designable, 0 otherwise",
            "1 if residue is T and not designable, 0 otherwise",
            "1 if residue is V and not designable, 0 otherwise",
            "1 if residue is W and not designable, 0 otherwise",
            "1 if residue is Y and not designable, 0 otherwise",
            "1 if residue is designable, 0 otherwise",
        ]

    def n_edge_features(self):
        return 0
