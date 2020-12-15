import numpy as np
import math

import menten_gcn as mg
from mg.decorators.standard import *
from mg.decorators.base import *
from mg.wrappers import *

#import scipy
#from scipy import sparse as sp

#import spektral
import tensorflow as tf
from tensorflow.keras.layers import Input
        
class DataMaker:

    '''
    TODO
    '''
    
    def __init__( self, decorators, edge_distance_cutoff_A, max_residues, exclude_bbdec = False, nbr_distance_cutoff_A = None ):
        '''
        TODO
        '''

        self.bare_bones_decorator = BareBonesDecorator()
        self.exclude_bbdec = exclude_bbdec
        if exclude_bbdec:
            decorators2 = []
        else:
            decorators2 = [ self.bare_bones_decorator ]
        decorators2.extend( decorators )
        self.all_decs = CombinedDecorator( decorators2 )
        
        self.edge_distance_cutoff_A = edge_distance_cutoff_A
        self.max_residues = max_residues
        if nbr_distance_cutoff_A == None:
            self.nbr_distance_cutoff_A = edge_distance_cutoff_A
        else:
            self.nbr_distance_cutoff_A = nbr_distance_cutoff_A

    def get_N_F_S( self ):
        '''
        TODO
        '''
        N = self.max_residues
        F = self.all_decs.n_node_features()
        S = self.all_decs.n_edge_features()
        return N, F, S

    def get_node_details( self ):
        '''
        TODO
        '''
        node_details = self.all_decs.describe_node_features()
        assert len( node_details ) == self.all_decs.n_node_features()
        return node_details

    def get_edge_details( self ):
        '''
        TODO
        '''
        edge_details = self.all_decs.describe_edge_features()
        assert len( edge_details ) == self.all_decs.n_edge_features()
        return edge_details
        
    def summary( self ):
        '''
        TODO
        '''
        node_details = self.get_node_details()
        edge_details = self.get_edge_details()

        print( "\nSummary:\n" )
            
        print( len(node_details), "Node Features:" )
        for i in range( 0, len(node_details) ):
            print( i+1, ":", node_details[i] )
            
        print( "" )
        
        print( len(edge_details), "Edge Features:" )
        for i in range( 0, len(edge_details) ):
            print( i+1, ":", edge_details[i] )

        print( "\n" )

        print( "This model can be reproduced by using these decorators:" )
        for i in self.all_decs.decorators:
            print( "-", i.get_version_name() )
        if not self.exclude_bbdec:
            print( "Note that the BareBonesDecorator is included by default and does not need to be explicitly provided" )

        print( "\nPlease cite: TODO\n" )

    def make_data_cache( self, wrapped_pose ):
        '''
        TODO
        '''
        cache = DecoratorDataCache( wrapped_pose )
        self.all_decs.cache_data( wrapped_pose, cache.dict_cache )
        return cache
        
    def _calc_nbrs( self, wrapped_pose, focused_resids, legal_nbrs=None ):
        '''
        TODO
        '''
        #includes focus in subset

        if legal_nbrs is None:
            legal_nbrs = wrapped_pose.get_legal_nbrs() #Still might be None
        
        focus_xyzs = []
        nbrs = []
        for fres in focused_resids:
            focus_xyzs.append( wrapped_pose.get_atom_xyz( fres, "CA" ) )
            nbrs.append( (-100.0,fres) )
            
        for resid in range( 1, wrapped_pose.n_residues() + 1 ):
            if (resid in focused_resids):
                continue
            if legal_nbrs is not None:
                if not legal_nbrs[ resid ]:
                    continue
            xyz = wrapped_pose.get_atom_xyz( resid, "CA" )
            min_dist = 99999.9
            for fxyz in focus_xyzs:
                min_dist = min( min_dist, np.linalg.norm( xyz - fxyz ) )
            if min_dist > self.nbr_distance_cutoff_A:
                continue
            nbrs.append( (min_dist, resid) )
            
        if len( nbrs ) > self.max_residues:
            #print( "AAA", len( nbrs ), self.max_residues )            
            nbrs = sorted(nbrs, key=lambda tup: tup[0])
            nbrs = nbrs[0:self.max_residues]
            assert len( nbrs ) == self.max_residues
            
        final_resids = []
        for n in nbrs:
            final_resids.append( n[1] )
        return final_resids

    def _get_edge_data_for_pair( self, wrapped_pose, resid_i, resid_j, data_cache=None ):
        '''
        TODO
        '''
        if data_cache.edge_cache is not None:
            if resid_j in data_cache.edge_cache[resid_i]:
                assert resid_i in data_cache.edge_cache[resid_j]
                return data_cache.edge_cache[resid_i][resid_j], data_cache.edge_cache[resid_j][resid_i]

        f_ij, f_ji = self.all_decs.calc_edge_features( wrapped_pose, resid1=resid_i, resid2=resid_j, dict_cache=data_cache.dict_cache )
        assert len( f_ij ) == self.all_decs.n_edge_features()
        assert len( f_ji ) == self.all_decs.n_edge_features()

        f_ij = np.asarray( f_ij )
        f_ji = np.asarray( f_ji )
        if data_cache.edge_cache is not None:
            data_cache.edge_cache[resid_i][resid_j] = f_ij
            data_cache.edge_cache[resid_j][resid_i] = f_ji
        return f_ij,f_ji
    
    def _calc_adjacency_matrix_and_edge_data( self, wrapped_pose, all_resids, data_cache=None ):
        '''
        TODO
        '''
        N, F, S = self.get_N_F_S()
        A_dense = np.zeros( shape=[N,N])
        E_dense = np.zeros( shape=[N,N,S])
        
        for i in range( 0, len(all_resids)-1 ):
            resid_i = all_resids[ i ]
            i_xyz = wrapped_pose.get_atom_xyz( resid_i, "CA" )
            for j in range( i + 1, len(all_resids) ):
                resid_j = all_resids[ j ]
                j_xyz = wrapped_pose.get_atom_xyz( resid_j, "CA" )
                dist = np.linalg.norm( i_xyz - j_xyz )
                if dist < self.edge_distance_cutoff_A:
                    f_ij, f_ji = self._get_edge_data_for_pair( wrapped_pose, resid_i=resid_i, resid_j=resid_j, data_cache=data_cache )
                    A_dense[ i ][ j ] = 1.0
                    E_dense[ i ][ j ] = f_ij
                    
                    A_dense[ j ][ i ] = 1.0
                    E_dense[ j ][ i ] = f_ji

        return A_dense, E_dense
    
    def _get_node_data( self, wrapped_pose, resids, data_cache ):
        '''
        TODO
        '''
        N, F, S = self.get_N_F_S()
        X = np.zeros( shape=[N,F] )
        index = -1
        for resid in resids:
            index += 1
            if data_cache.node_cache is not None:
                if data_cache.node_cache[ resid ] is not None:
                    X[ index ] = data_cache.node_cache[ resid ]
                    if not self.exclude_bbdec:
                        #Redo focus residues
                        new_bbdec = self.bare_bones_decorator.calc_node_features( wrapped_pose, resid )
                        assert len( new_bbdec ) == 1
                        X[ index ][ 0 ] = new_bbdec[ 0 ]
                        #print( "REDO", index, resid, new_bbdec[ 0 ] )
                        #print( X )
                    continue

            n = self.all_decs.calc_node_features( wrapped_pose, resid )
            
            n = np.asarray( n )
            if data_cache.node_cache is not None:
                data_cache.node_cache[ resid ] = n
            X[ index ] = n
        if not self.exclude_bbdec:
            #assumes at least one focus resid
            if X[ 0 ][ 0 ] != 1:
                print( "Error: X[ 0 ][ 0 ] == ", X[ 0 ][ 0 ] )
                for i in range( 0, len(resids) ):
                    print( i, resids[i], X[ i ][ 0 ] )
            assert X[ 0 ][ 0 ] == 1
        return X
    
    def generate_XAE_input_tensors( self ):
        '''
        TODO
        '''
        N, F, S = self.get_N_F_S()
        X_in = Input( shape=(N,F), name='X_in')
        A_in = Input( shape=(N,N), sparse=False, name='A_in')
        E_in = Input( shape=(N,N,S), name='E_in')
        return X_in, A_in, E_in
    
    def generate_input( self, wrapped_pose, focused_resids, data_cache=None, legal_nbrs=None ):
        '''
        TODO
        '''
        if data_cache is None:
            data_cache = NullDecoratorDataCache()
            
        self.bare_bones_decorator.set_focused_resids( focused_resids )
        all_resids = self._calc_nbrs( wrapped_pose, focused_resids, legal_nbrs=legal_nbrs )
        #all_resids.sort() #Why???
        n_nodes = len( all_resids )
        
        # Node Data
        X = self._get_node_data( wrapped_pose, all_resids, data_cache )
        
        # Adjacency Matrix and Edge Data
        A, E = self._calc_adjacency_matrix_and_edge_data( wrapped_pose, all_resids, data_cache=data_cache )

        #TODO reorder
        #TODO replace NAN with, what, -10?
        return X, A, E, all_resids

    def generate_input_for_resid( self, wrapped_pose, resid, data_cache=None, legal_nbrs=None ):
        '''
        TODO
        '''
        return self.generate_input( wrapped_pose, focused_resids=[resid], data_cache=data_cache, legal_nbrs=legal_nbrs )
