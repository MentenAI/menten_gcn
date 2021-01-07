import spektral
import tensorflow as tf
import numpy as np
import math
import random
import gc

from typing import List
from menten_gcn.wrappers import WrappedPose


class DecoratorDataCache:

    """
    DecoratorDataCache prevents re-calculating the same node/edge data many times.
    You will need to create a different cache for each pose you work with.

    Also, we highly recommend you make this inside the DataMaker (calling data_maker.make_data_cache() ).
    This allows for further caching and speedups.

    Parameters
    ----------
    wrapped_pose: WrappedPose
        Please pass the pose that we should make a cache for
    """

    def __init__(self, wrapped_pose: WrappedPose):
        # lookup is edge_cache[i][j]
        self.edge_cache = [dict() for x in range(wrapped_pose.n_residues() + 1)]
        self.node_cache = [None for x in range(wrapped_pose.n_residues() + 1)]
        self.dict_cache = dict()


class NullDecoratorDataCache:
    def __init__(self):
        self.edge_cache = None
        self.node_cache = None
        self.dict_cache = None


class DataHolder:

    """
    DataHolder is a wonderful class that automatically stores the direct output of the DataMaker.
    The DataHolder can then feed your data directly into kera's model.fit() method using the generators below.

    There are descriptions for each method below but perhaps the best way to grasp
    the DataHolder's usage is to see the example at the bottom.
    """

    def __init__(self):
        self.Xs = []
        self.As = []
        self.Es = []
        self.outs = []

    def assert_mode(self, mode=spektral.layers.ops.modes.BATCH):
        """
        For those of you using spektral, this ensures that your data is in the correct shape.
        Unfortunately this only currently checks X and A.
        More development is incoming
        """

        if len(self.Xs) == 0:
            raise RuntimeError("DataHolder.assert_mode is called before any data is added")
        tf_As = tf.convert_to_tensor(np.asarray(self.As))
        tf_Xs = tf.convert_to_tensor(np.asarray(self.Xs))
        assert spektral.layers.ops.modes.autodetect_mode(tf_As, tf_Xs) == mode

    def append(self, X: np.ndarray, A: np.ndarray, E: np.ndarray, out: np.ndarray):
        """
        This is the most important method in this class:
        it gives the data to the dataholder.

        Parameters
        ----------
        X: array-like
            Node features, shape=(N,F)
        A: array-like
            Adjacency Matrix, shape=(N,N)
        E: array-like
            Edge features, shape=(N,N,S)
        out: array-like
            What is the output of your model supposed to be? You decide the shape.
        """

        # TODO assert shape
        self.Xs.append(np.asarray(X))
        self.As.append(np.asarray(A))
        self.Es.append(np.asarray(E))
        self.outs.append(np.asarray(out))

    def size(self) -> int:
        return len(self.Xs)

    def get_batch(self, begin: int, end: int):
        assert begin >= 0
        assert end <= self.size()
        x = np.asarray(self.Xs[begin:end])
        a = np.asarray(self.As[begin:end])
        e = np.asarray(self.Es[begin:end])
        o = np.asarray(self.outs[begin:end])
        # TODO debug mode
        # for xi in x:
        #    assert xi.flatten()[ 0 ] == 1
        return [x, a, e], o

    def get_indices(self, inds):
        """
        this stopped working at some point
        x = np.asarray(self.Xs[inds])
        a = np.asarray(self.As[inds])
        e = np.asarray(self.Es[inds])
        o = np.asarray(self.outs[inds])
        """

        x = np.asarray([self.Xs[i] for i in inds])
        a = np.asarray([self.As[i] for i in inds])
        e = np.asarray([self.Es[i] for i in inds])
        o = np.asarray([self.outs[i] for i in inds])

        # TODO debug mode
        # for xi in x:
        #    assert xi.flatten()[ 0 ] == 1
        return [x, a, e], o

    def save_to_file(self, fileprefix: str):
        """
        Want to save this data for later?
        Use this method to cache it to disk.

        Users of this method may be interested in the CachedDataHolderInputGenerator below

        Parameters
        ----------
        fileprefix: str
            Filename prefix for cache.
            fileprefix="foo/bar" will result in creating "./foo/bar.npz"
        """
        np.savez_compressed(
            fileprefix + '.npz',
            x=np.asarray(
                self.Xs),
            a=np.asarray(
                self.As),
            e=np.asarray(
                self.Es),
            o=np.asarray(
                self.outs))

    def load_from_file(self, fileprefix: str = None, filename: str = None):
        """
        save_to_file's partner. Use this to load in caches already saved.
        Please provide either fileprefix or filename, but not both.

        This duplicity may seem silly. The goal for fileprefix is to be consistant with save_to_file
        (the two "fileprefix" args will be identical strings for both)
        whereas the goal for filename is to simply list the name of the file verbosely.

        Parameters
        ----------
        fileprefix: str
            Filename prefix for cache.
            fileprefix="foo/bar" will result in reading "./foo/bar.npz"
        filename: str
            Filename for cache.
            fileprefix="foo/bar.npz" will result in reading "./foo/bar.npz"
        """

        assert filename is None or fileprefix is None, "Please provide either fileprefix or filename"
        assert filename is not None or fileprefix is not None, "Please provide either fileprefix or filename"

        if filename is None:
            fn = fileprefix + '.npz'
        else:
            fn = filename

        cache = np.load(fn)
        self.Xs = cache['x']
        self.As = cache['a']
        self.Es = cache['e']
        self.outs = cache['o']
        assert not np.isnan(np.sum(self.Xs)), filename
        assert not np.isnan(np.sum(self.As)), filename
        assert not np.isnan(np.sum(self.Es)), filename
        assert not np.isnan(np.sum(self.outs)), filename


class DataHolderInputGenerator(tf.keras.utils.Sequence):

    """
    This class is used to feed a DataHolder directly into
    Keras's model.fit() protocol. See the example code below.

    Parameters
    ----------
    data_holder: DataHolder
        A DataHolder that you just made
    batch_size: int
        How many elements should be grouped together
        in batches during training?
    """

    def __init__(self, data_holder: DataHolder, batch_size: int = 32):
        self.holder = data_holder
        self.batch_size = batch_size
        self.indices = [i for i in range(0, data_holder.size())]

    def n_elem(self) -> int:
        return self.holder.size()

    def __len__(self):
        return int((self.holder.size() + self.batch_size - 1) / self.batch_size)

    def __getitem__(self, item_index):
        begin = item_index * self.batch_size
        end = min(begin + self.batch_size, len(self.indices))

        inds = self.indices[begin:end]
        inp, out = self.holder.get_indices(inds)

        for i in inp:
            assert np.isfinite(i).all()
        assert np.isfinite(out).all()
        return inp, out

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        gc.collect()


class CachedDataHolderInputGenerator(tf.keras.utils.Sequence):

    """
    This class is used to feed a DataHolder directly into
    Keras's model.fit() protocol.

    The difference with this class is that it reads one or more DataHolders
    that have been saved onto disk.

    See the example code below.

    Parameters
    ----------
    data_list_lines: list
        A list of filenames, each one for a different DataHolder.
    cache: bool
        If true, this class will load every DataHolder
        into memory once and keep them there.
        This can require a lot of memory.
        Otherwise, we will only read in one DataHolder at a time
        (once per epoch).
        This increases disk IO but is often worth it.
    batch_size: int
        How many elements should be grouped together
        in batches during training?
    autoshuffle: bool
        This is very nuanced so we recommend keeping the default value of None
        (this lets us pick the appropriate action).
        Long story short: YOU DO NOT WANT TO DO SHUFFLE=TRUE inside keras's
        model.fit() when cache=False because disk IO goes through the roof.
        To counter this, we handle shuffling internally
        in a way that minimizes disk IO.
        However you DO WANT TO DO SHUFFLE=TRUE if cache=True
        because everything is in memory anyways.
        I know this is confusing.
        Maybe this will be cleaner in the future.
    """

    def __init__(self, data_list_lines: List[str], cache: bool = False, batch_size: int = 32, autoshuffle: bool = None):
        print("Generating from", str(len(data_list_lines)), "files")
        self.data_list_lines = data_list_lines

        if autoshuffle is None:
            self.autoshuffle = not cache
        else:
            self.autoshuffle = autoshuffle
            assert not(self.autoshuffle and cache), "Autoshuffle is not compatible with caching yet."

        self.cache = cache
        self.cached_data = [None for i in self.data_list_lines]

        self.batch_size = batch_size

        self.sizes = []
        self.total_size = 0

        if not self.cache:
            self.indices = []
        else:
            self.indices = None

        for i in range(0, len(self.data_list_lines)):
            filename = self.data_list_lines[i]
            holder = DataHolder()
            holder.load_from_file(filename=filename)
            size = holder.size()
            size = (int(math.floor(size / float(self.batch_size))) * self.batch_size)
            print("rounding", holder.size(), "to", size)
            # round DOWN to nearest multiple of batch size
            self.sizes.append(size)
            self.total_size += size
            if self.cache:
                self.cached_data[i] = holder
            else:
                del holder
            gc.collect()
        print("    ", self.total_size, "elements")

        self.sizes = np.asarray(self.sizes)
        self.cum_sizes = np.cumsum(self.sizes)

        self.currently_loaded_npz_index = -1

    """
    def n_elem(self):
        return len(self.data_list_lines)
    """

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return int(self.total_size / self.batch_size)

    def get_npz_index_for_item(self, item_index):
        resized_i = item_index * self.batch_size
        for i in range(0, len(self.cum_sizes)):
            if resized_i < self.cum_sizes[i]:
                if i == 0:
                    return i, item_index
                else:
                    return i, int(item_index - (self.cum_sizes[i - 1] / self.batch_size))
        assert False, "DEAD CODE IN get_npz_index_for_item"

    def __getitem__(self, item_index):
        npz_i, i = self.get_npz_index_for_item(item_index)
        if self.cache:
            self.holder = self.cached_data[npz_i]
        elif npz_i != self.currently_loaded_npz_index:
            self.holder = DataHolder()
            gc.collect()
            self.currently_loaded_npz_index = npz_i
            self.holder.load_from_file(filename=self.data_list_lines[self.currently_loaded_npz_index])
            self.indices = [x for x in range(0, self.holder.size())]
            if self.autoshuffle:
                np.random.shuffle(self.indices)

        begin = i * self.batch_size
        end = min(begin + self.batch_size, len(self.holder.As))
        if self.indices is None:
            inp, out = self.holder.get_batch(begin, end)
        else:
            assert end <= len(self.indices)
            inds = self.indices[begin:end]
            inp, out = self.holder.get_indices(inds)

        for i in inp:
            assert np.isfinite(i).all()
        assert np.isfinite(out).all()
        return inp, out

    def on_epoch_end(self):
        if self.autoshuffle:
            self.shuffle()
        gc.collect()

    def shuffle(self):
        # https://www.geeksforgeeks.org/python-shuffle-two-lists-with-same-order/
        # TODO: get this to work with cached data
        assert not self.cache

        # shuffle
        temp = list(zip(self.data_list_lines, self.sizes))
        random.shuffle(temp)
        self.data_list_lines, self.sizes = zip(*temp)

        # recalc
        self.cum_sizes = np.cumsum(self.sizes)

        # reset
        self.holder = None
        self.currently_loaded_npz_index = -1
