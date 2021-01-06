import math

from menten_gcn.decorators.base import Decorator
# from menten_gcn.decorators.geometry import *
# from menten_gcn.decorators.sequence import *


class BareBonesDecorator(Decorator):

    """
    This decorator is included in all DataMakers by default.
    Its goal is to be the starting point upon which everything else is built.
    It labels focus nodes and labels edges for residues that are polymer bonded to one another.

    - 1 Node Feature
    - 1 Edge Feature
    """

    def __init__(self):
        self.focused_resids = []

    def get_version_name(self):
        return "BareBonesDecorator"

    def set_focused_resids(self, focused_resids):
        self.focused_resids = focused_resids

    def n_node_features(self):
        # 0: is focus residue
        return 1

    def calc_node_features(self, wrapped_pose, resid, dict_cache=None):
        if resid in self.focused_resids:
            return [1.0]
        else:
            return [0.0]

    def describe_node_features(self):
        return [
            "1 if the node is a focus residue, 0 otherwise",
        ]

    def n_edge_features(self):
        # 0: polymer bond
        return 1

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        result1 = 0.0
        if wrapped_pose.residues_are_polymer_bonded(resid1, resid2):
            result1 = 1.0
        f = [result1]
        return f, f

    def describe_edge_features(self):
        return [
            "1.0 if the two residues are polymer-bonded, 0.0 otherwise",
        ]


class SequenceSeparation(Decorator):

    """
    The sequence distance between the two residues
    (i.e., number of residues between these two residues in sequence space, plus one).
    -1.0 if the two residues belong to different chains.

    - 0 Node Features
    - 1 Edge Feature

    Parameters
    ---------
    ln: bool
        Report the natural log of the distance instead of the raw count. Does not apply to -1 values
    """

    def __init__(self, ln: bool = True):
        self.ln = ln

    def get_version_name(self):
        return "SequenceSeparation"

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        return 1

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        if not wrapped_pose.resids_are_same_chain(resid1, resid2):
            return [-1.0, ], [-1.0]
        distance = abs(resid1 - resid2)
        assert distance >= 0
        if self.ln:
            distance = math.log(distance)
        return [distance, ], [distance, ]

    def describe_edge_features(self):
        if self.ln:
            return [
                "Natural Log of the sequence distance between the two residues " +
                "(i.e., number of residues between these two residues in sequence space, plus one). " +
                "-1.0 if the two residues belong to different chains. (symmetric)",
            ]
        else:
            return [
                "The sequence distance between the two residues " +
                "(i.e., number of residues between these two residues in sequence space, plus one). " +
                "-1.0 if the two residues belong to different chains. (symmetric)",
            ]


class SameChain(Decorator):

    """
    1 if the two residues are part of the same protein chain. Otherwise 0.

    - 0 Node Features
    - 1 Edge Feature
    """

    def get_version_name(self):
        return "SameChain"

    def n_node_features(self):
        return 0

    def n_edge_features(self):
        return 1

    def calc_edge_features(self, wrapped_pose, resid1, resid2, dict_cache=None):
        if wrapped_pose.resids_are_same_chain(resid1, resid2):
            return [1.0, ], [1.0]
        else:
            return [0.0, ], [0.0, ]

    def describe_edge_features(self):
        return ["1 if the two residues belong to the same chain, otherwise 0. (symmetric)", ]
