import spektral
import tensorflow as tf
import numpy as np
import math
import random
import gc

class DataHolder:
    def __init__( self, draft = None ):
        self.Xs = []
        self.As = []        
        self.Es = []
        self.outs = []
        #TODO store NFS

    def assert_mode( self, mode = spektral.layers.ops.modes.BATCH ):
        if len(self.Xs) == 0:
            raise RuntimeError( "DataGator.assert_mode is called before any data is added" )
        tf_As = tf.convert_to_tensor( np.asarray( self.As ) )
        tf_Xs = tf.convert_to_tensor( np.asarray( self.Xs ) )
        assert spektral.layers.ops.modes.autodetect_mode(tf_As,tf_Xs) == mode

        
    def append( self, X, A, E, out ):
        #TODO assert shape
        self.Xs.append( np.asarray(X) )
        self.As.append( np.asarray(A) )
        self.Es.append( np.asarray(E) )
        self.outs.append( np.asarray(out) )

    def size( self ):
        return len( self.Xs )

    def get_batch( self, begin, end ):
        assert begin >= 0
        assert end <= self.size()
        x = np.asarray( self.Xs[begin:end] )
        a = np.asarray( self.As[begin:end] )
        e = np.asarray( self.Es[begin:end] )
        o = np.asarray( self.outs[begin:end] )
        #TODO debug mode
        #for xi in x:
        #    assert xi.flatten()[ 0 ] == 1
        return [x,a,e], o

    def get_indices( self, inds ):
        x = np.asarray( self.Xs[inds] )
        a = np.asarray( self.As[inds] )
        e = np.asarray( self.Es[inds] )
        o = np.asarray( self.outs[inds] )
        #TODO debug mode
        #for xi in x:
        #    assert xi.flatten()[ 0 ] == 1
        return [x,a,e], o
    
    
    def memory_usage_MiB( self ):
        #I think this is wrong
        amount = self.Xs.__sizeof__() + self.As.__sizeof__() + self.Es.__sizeof__() + self.outs.__sizeof__()
        return amount / (1024*1024)

    def save_to_file( self, fileprefix ):
        np.savez_compressed( fileprefix + '.npz', x=np.asarray( self.Xs), a=np.asarray( self.As), e=np.asarray( self.Es), o=np.asarray( self.outs ) )

    def load_from_file( self, fileprefix = None, filename = None ):
        assert filename == None or fileprefix == None
        assert filename != None or fileprefix != None        

        if filename == None:
            fn = fileprefix + '.npz'
        else:
            fn = filename
            
        cache = np.load( fn )
        self.Xs = cache[ 'x' ]
        self.As = cache[ 'a' ]
        self.Es = cache[ 'e' ]
        self.outs = cache[ 'o' ]
        assert not np.isnan(np.sum( self.Xs )), filename
        assert not np.isnan(np.sum( self.As )), filename
        assert not np.isnan(np.sum( self.Es )), filename
        assert not np.isnan(np.sum( self.outs )), filename        

class CachedDataHolderInputGenerator( tf.keras.utils.Sequence ):
    
    def __init__(self, data_list_lines, cache=False, batch_size=32, autoshuffle=None):
        print( "Generating from", str(len(data_list_lines)), "files" )
        self.data_list_lines = data_list_lines

        if autoshuffle == None:
            self.autoshuffle = not cache
        else:
            self.autoshuffle = autoshuffle
            assert not( self.autoshuffle and cache ), "Autoshuffle is not compatible with caching yet."
            
        self.cache = cache
        self.cached_data = [ None for i in self.data_list_lines ]
            
        self.batch_size = batch_size

        self.sizes = []
        self.total_size = 0

        if not self.cache:
            self.indices = []
        else:
            self.indices = None
            
        for i in range( 0, len( self.data_list_lines ) ):
            filename = self.data_list_lines[ i ]
            gator = DataGator()
            gator.load_from_file( filename=filename )
            size = gator.size()
            size = ( int(math.floor(size / float(self.batch_size))) * self.batch_size )
            print( "rounding", gator.size(), "to", size )
            #round DOWN to nearest multiple of batch size
            self.sizes.append( size )
            self.total_size += size
            if self.cache:
                self.cached_data[ i ] = gator
            else:
                del gator
            gc.collect()
        print( "    ", self.total_size, "elements" )

        self.sizes = np.asarray( self.sizes )
        self.cum_sizes = np.cumsum( self.sizes )

        self.currently_loaded_npz_index = -1
        
    def n_elem( self ):
        return len(self.data_list_lines)

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        #return int(len( self.gator.Xs ) / 100)
        return int(self.total_size / self.batch_size)

    def get_npz_index_for_item( self, item_index ):
        resized_i = item_index * self.batch_size
        for i in range( 0, len(self.cum_sizes) ):
            if resized_i < self.cum_sizes[i]:
                if i == 0:
                    return i, item_index
                else:
                    return i, int(item_index - (self.cum_sizes[i-1] / self.batch_size))
        assert False, "DEAD CODE IN get_npz_index_for_item"

    def __getitem__(self, item_index):
        npz_i, i = self.get_npz_index_for_item( item_index )
        if self.cache:
            self.gator = self.cached_data[ npz_i ]
        elif npz_i != self.currently_loaded_npz_index:
            self.gator = DataGator()
            gc.collect()
            self.currently_loaded_npz_index = npz_i
            self.gator.load_from_file( filename=self.data_list_lines[ self.currently_loaded_npz_index ] )
            self.indices = [ x for x in range( 0, self.gator.size() ) ]
            if self.autoshuffle:
                np.random.shuffle( self.indices )

        begin = i * self.batch_size
        end = min( i + self.batch_size, len( self.gator.As ) )
        if self.indices == None:
            inp, out = self.gator.get_batch( begin, end )
        else:
            assert end <= len(self.indices)
            inds = self.indices[begin:end]
            inp,out = self.gator.get_indices(inds)
            
        for i in inp:
            assert np.isfinite( i ).all()
        assert np.isfinite( out ).all()            
        return inp, out

    def on_epoch_end(self):
        if self.autoshuffle:
            self.shuffle()
        gc.collect()

    def shuffle(self):
        #https://www.geeksforgeeks.org/python-shuffle-two-lists-with-same-order/
        #TODO: get this to work with cached data
        assert not self.cache

        #shuffle
        temp = list(zip(self.data_list_lines,self.sizes))
        random.shuffle(temp)
        self.data_list_lines,self.sizes = zip(*temp)

        #recalc
        self.cum_sizes = np.cumsum( self.sizes )

        #reset
        self.gator = None
        self.currently_loaded_npz_index = -1        
        
