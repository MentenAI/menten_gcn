import numpy as np
import math

import menten_gcn as mg
from mg.decorators.base import Decorator
from mg.decorators.geometry import *
from mg.decorators.sequence import *

class BareBonesDecorator( Decorator ):

    def __init__( self ):
        self.focused_resids = []
    
    def get_version_name( self ):
        return "BareBonesDecorator"
    
    def set_focused_resids( self, focused_resids ):
        self.focused_resids = focused_resids
    
    def n_node_features( self ):
        #0: is focus residue
        return 1

    def calc_node_features( self, wrapped_protein, resid, dict_cache=None ):
        if resid in self.focused_resids:
            return [ 1.0 ]
        else:
            return [ 0.0 ]

    def describe_node_features( self ):
        return [
            "1 if the node is a focus residue, 0 otherwise",
            ]
    
    def n_edge_features( self ):
        #0: polymer bond
        return 1

    def calc_edge_features( self, wrapped_protein, resid1, resid2, dict_cache=None ):
        result1 = 0.0
        if wrapped_protein.residues_are_polymer_bonded( resid1, resid2 ):
            result1 = 1.0
        f = [ result1 ]
        return f, f

    def describe_edge_features( self ):
        return [
            "1.0 if the two residues are polymer-bonded, 0.0 otherwise",
            ]

class SequenceSeparation( Decorator ):
    def __init__( self, ln=True ):
        self.ln = ln
    
    def get_version_name( self ):
        return "SequenceSeparation"
        
    def n_node_features( self ):
        return 0
    
    def n_edge_features( self ):
        return 1

    def calc_edge_features( self, wrapped_protein, resid1, resid2, dict_cache=None ):
        if not wrapped_protein.resids_are_same_chain( resid1, resid2 ):
            return [-1.0,], [-1.0]
        distance = abs( resid1 - resid2 )
        assert distance >= 0
        if self.ln:
            distance = math.log( distance )
        return [distance,], [distance,]

    def describe_edge_features( self ):
        if self.ln:
            return [ "Natural Log of the sequence distance between the two residues (i.e., number of residues between these two residues in sequence space, plus one). -1.0 if the two residues belong to different chains. (symmetric)", ]
        else:
            return [ "The sequence distance between the two residues (i.e., number of residues between these two residues in sequence space, plus one). -1.0 if the two residues belong to different chains. (symmetric)", ]
    
class SameChain( Decorator ):
    
    def get_version_name( self ):
        return "SameChain"
        
    def n_node_features( self ):
        return 0
    
    def n_edge_features( self ):
        return 1

    def calc_edge_features( self, wrapped_protein, resid1, resid2, dict_cache=None ):
        if wrapped_protein.resids_are_same_chain( resid1, resid2 ):
            return [1.0,], [1.0]
        else:
            return [0.0,], [0.0,]

    def describe_edge_features( self ):
            return [ "1 if the two residues belong to the same chain, otherwise 0. (symmetric)", ]
    