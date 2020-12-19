#import numpy as np
#import math

from menten_gcn.wrappers import *
from menten_gcn.data_management import *


class Decorator:

    #########
    # BASIC #
    #########

    def __init__(self):
        pass

    def get_version_name(self):
        raise NotImplementedError

    def cache_data(self, wrapped_pose, dict_cache):
        pass

    #########
    # NODES #
    #########

    def n_node_features(self):
        return 0

    def calc_node_features(self, wrapped_protein, resid, dict_cache=None):
        features = []
        return features

    def describe_node_features(self):
        # Describe in reproducible detail how these numbers are calculated
        return []

    #########
    # EDGES #
    #########

    def n_edge_features(self):
        return 0

    def calc_edge_features(self, wrapped_protein, resid1, resid2, dict_cache=None):
        features = []  # 1 -> 2
        inv_features = []  # 2 -> 1
        return features, inv_features

    def describe_edge_features(self):
        # Describe in reproducible detail how these numbers are calculated
        return []


class CombinedDecorator(Decorator):

    #########
    # BASIC #
    #########

    def __init__(self, decorators=[]):
        self.decorators = decorators

    def get_version_name(self):
        name = "CombinedDecorator("
        for d in self.decorators:
            name += d.get_version_name() + ","
        name += ")"
        return name

    def cache_data(self, wrapped_pose, dict_cache):
        for d in self.decorators:
            d.cache_data(wrapped_pose, dict_cache)

    #########
    # NODES #
    #########

    def n_node_features(self):
        return sum(d.n_node_features() for d in self.decorators)

    def calc_node_features(self, pose, resid, dict_cache=None):
        features = []
        for d in self.decorators:
            features.extend(d.calc_node_features(pose, resid=resid, dict_cache=dict_cache))
        '''
        for d in self.decorators:
            print( d.get_version_name(), d.n_node_features(), len( d.calc_node_features( pose, resid=resid, dict_cache=dict_cache ) ) )
        assert( len( features ) == self.n_node_features() )
        '''
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

    def calc_edge_features(self, pose, resid1, resid2, dict_cache=None):
        features = []  # 1 -> 2
        inv_features = []  # 2 -> 1
        for d in self.decorators:
            f12, f21 = d.calc_edge_features(pose, resid1=resid1, resid2=resid2, dict_cache=dict_cache)
            features.extend(f12)
            inv_features.extend(f21)
        assert(len(features) == self.n_edge_features())
        assert(len(features) == len(inv_features))
        return features, inv_features

    def describe_edge_features(self):
        features = []
        for d in self.decorators:
            features.extend(d.describe_edge_features())
        '''
        print( len( features ), self.n_edge_features() )
        for e in features:
            print( e )
        '''
        assert(len(features) == self.n_edge_features())
        return features
