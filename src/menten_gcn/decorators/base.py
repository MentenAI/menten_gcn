from menten_gcn.wrappers import WrappedPose
from typing import List, Tuple


class Decorator:

    #########
    # BASIC #
    #########

    def __init__(self):
        pass

    def get_version_name(self) -> str:
        """
        Get a unique, versioned name of this decorator for maximal reproducability
        """

        raise NotImplementedError

    def cache_data(self, wrapped_pose: WrappedPose, dict_cache: dict):
        """
        Some decorators can save time by precomputing arbitrary data
        and storing it in this cache.
        For example, the RosettaHBondDecorator recomputes and caches
        all hydrogen bonds so they become a simple lookup when decorating
        individual nodes and edges.

        Parameters
        ---------
        wrapped_pose: WrappedPose
            Each pose will be given its own cache.
            This pose is the one we are currently caching
        dict_cache: dict
            Destination for your data.
            Please use a unique key that won't overlap with other decorators'.
        """
        pass

    #########
    # NODES #
    #########

    def n_node_features(self) -> int:
        """
        How many features will this decorator add to node tensors (X)?
        """
        return 0

    def calc_node_features(self, wrapped_pose: WrappedPose,
                           resid: int, dict_cache: dict = None) -> List[float]:
        """
        This does all of the business logic of calculating
        the values to be added for each node.

        Parameters
        ---------
        wrapped_pose: WrappedPose
            The pose we are currently generating data for
        resid: int
            The residue ID we are currently generating data for
        dict_cache: dict
            The same cache that was populated in "cache_data".
            The user might not have created a cache so don't assume this is not None.
            See the RosettaHBondDecorator for an example of how to use this

        Returns
        ---------
        features: list
            The length of this list will be the same value as self.n_node_features().
            These are the values to represent this decorator's
            contribution to X for this resid.
        """
        features = []
        return features

    def describe_node_features(self) -> List[str]:
        """
        Returns descriptions of how each value is computed.
        Our goal is for these descriptions to be relatively concise but
        also have enough detail to fully reproduce these calculations.

        Returns
        ---------
        features: list
            The length of this list will be the same value as self.n_node_features().
            These are descriptions of the values to represent this decorator's
            contribution to X for any arbitrary resid.
        """
        return []

    #########
    # EDGES #
    #########

    def n_edge_features(self) -> int:
        """
        How many features will this decorator add to edge tensors (E)?
        """
        return 0

    def calc_edge_features(self, wrapped_pose: WrappedPose, resid1: int, resid2: int,
                           dict_cache: dict = None) -> Tuple[List[float], List[float]]:
        """
        This does all of the business logic of calculating
        the values to be added for each edge.

        This function will never be called in the reverse order
        (with resid1 and resid2 swapped).
        Instead, we just create both edges at once.

        Parameters
        ---------
        wrapped_pose: WrappedPose
            The pose we are currently generating data for
        resid1: int
            The first residue ID we are currently generating data for
        resid1: int
            The second residue ID we are currently generating data for
        dict_cache: dict
            The same cache that was populated in "cache_data".
            The user might not have created a cache so don't assume this is not None.
            See the RosettaHBondDecorator for an example of how to use this

        Returns
        ---------
        features: list
            The length of this list will be the same value as self.n_edge_features().
            These are the values to represent this decorator's
            contribution to E for the edge going from resid1 -> resid2.
        inv_features: list
            The length of this list will be the same value as self.n_edge_features().
            These are the values to represent this decorator's
            contribution to E for the edge going from resid2 -> resid1.
        """

        features = []  # 1 -> 2
        inv_features = []  # 2 -> 1
        return features, inv_features

    def describe_edge_features(self) -> List[str]:
        """
        Returns descriptions of how each value is computed.
        Our goal is for these descriptions to be relatively concise but
        also have enough detail to fully reproduce these calculations.

        Returns
        ---------
        features: list
            The length of this list will be the same value as self.n_edge_features().
            These are descriptions of the values to represent this decorator's
            contribution to E for any arbitrary resid pair.
        """
        return []


class CombinedDecorator(Decorator):

    #########
    # BASIC #
    #########

    def __init__(self, decorators: list = []):
        self.decorators = decorators

    def get_version_name(self):
        name = "CombinedDecorator("
        for d in self.decorators:
            name += d.get_version_name() + ","
        name += ")"
        return name

    def cache_data(self, wrapped_pose: WrappedPose, dict_cache: dict):
        for d in self.decorators:
            d.cache_data(wrapped_pose, dict_cache)

    #########
    # NODES #
    #########

    def n_node_features(self):
        return sum(d.n_node_features() for d in self.decorators)

    def calc_node_features(self, wrapped_pose: WrappedPose, resid: int, dict_cache: dict = None):
        features = []
        for d in self.decorators:
            features.extend(d.calc_node_features(wrapped_pose, resid=resid, dict_cache=dict_cache))
        assert(len(features) == self.n_node_features())
        return features

    def describe_node_features(self):
        features = []
        for d in self.decorators:
            features.extend(d.describe_node_features())
        assert(len(features) == self.n_node_features())
        return features

    #########
    # EDGES #
    #########

    def n_edge_features(self):
        return sum(d.n_edge_features() for d in self.decorators)

    def calc_edge_features(self, wrapped_pose: WrappedPose,
                           resid1: int, resid2: int, dict_cache: dict = None):
        features = []  # 1 -> 2
        inv_features = []  # 2 -> 1
        for d in self.decorators:
            f12, f21 = d.calc_edge_features(wrapped_pose,
                                            resid1=resid1, resid2=resid2, dict_cache=dict_cache)
            features.extend(f12)
            inv_features.extend(f21)
        assert(len(features) == self.n_edge_features())
        assert(len(features) == len(inv_features))
        return features, inv_features

    def describe_edge_features(self):
        features = []
        for d in self.decorators:
            features.extend(d.describe_edge_features())
        assert(len(features) == self.n_edge_features())
        return features
