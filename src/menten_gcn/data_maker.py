import numpy as np

from menten_gcn.decorators import *
from menten_gcn.wrappers import WrappedPose
from menten_gcn.data_management import DecoratorDataCache, NullDecoratorDataCache

#import tensorflow as tf
from tensorflow.keras.layers import Input, Layer

from typing import List, Tuple


class DataMaker:

    """
    The DataMaker is the user's interface to controling the size and composition of their graph.

    Parameters
    ----------
    decorators: list
        List of decorators that you want to include
    edge_distance_cutoff_A: float
        An edge will be created between any two pairs of residues if their
        C-alpha atoms are within this distance (measured in Angstroms)
    max_residues: int
        What is the maximum number of nodes a graph can have?
        This includes focus and neighbor nodes.
        If the number of focus+neighbors exceeds this number, we will leave out the neighbors that are farthest away in 3D space.
    exclude_bbdec: bool
        Every DataMaker has a standard "bare bones" decorator that is prepended to the list of decorators you provide.
        Set this to false to remove it entirely.
    nbr_distance_cutoff_A: float
        A node will be included in the graph if it is within this distance (Angstroms) of any focus node.
        A value of None will set this equal to edge_distance_cutoff_A
    """

    def __init__(self, decorators: List[Decorator], edge_distance_cutoff_A: float, max_residues: int,
                 exclude_bbdec: bool = False, nbr_distance_cutoff_A: float = None):

        self.bare_bones_decorator = BareBonesDecorator()
        self.exclude_bbdec = exclude_bbdec
        if exclude_bbdec:
            decorators2 = []
        else:
            decorators2 = [self.bare_bones_decorator]
        decorators2.extend(decorators)
        self.all_decs = CombinedDecorator(decorators2)

        self.edge_distance_cutoff_A = edge_distance_cutoff_A
        self.max_residues = max_residues
        if nbr_distance_cutoff_A is None:
            self.nbr_distance_cutoff_A = edge_distance_cutoff_A
        else:
            self.nbr_distance_cutoff_A = nbr_distance_cutoff_A

    def get_N_F_S(self) -> Tuple[int, int, int]:
        """
        Returns
        ----------
        N: int
            Maximum number of nodes in the graph
        F: int
            Number of features for each node
        S: int
            Number of features for each edge
        """

        N = self.max_residues
        F = self.all_decs.n_node_features()
        S = self.all_decs.n_edge_features()
        return N, F, S

    def get_node_details(self) -> List[str]:
        node_details = self.all_decs.describe_node_features()
        assert len(node_details) == self.all_decs.n_node_features()
        return node_details

    def get_edge_details(self) -> List[str]:
        edge_details = self.all_decs.describe_edge_features()
        assert len(edge_details) == self.all_decs.n_edge_features()
        return edge_details

    def summary(self):
        """
        Print a summary of the graph decorations to console.
        The goal of this summary is to describe every feature with enough detail to be able to be reproduced externally.
        This will also print any relevant citation information for individual decorators.
        """

        node_details = self.get_node_details()
        edge_details = self.get_edge_details()

        print("\nSummary:\n")

        print(len(node_details), "Node Features:")
        for i in range(0, len(node_details)):
            print(i + 1, ":", node_details[i])

        print("")

        print(len(edge_details), "Edge Features:")
        for i in range(0, len(edge_details)):
            print(i + 1, ":", edge_details[i])

        print("\n")

        print("This model can be reproduced by using these decorators:")
        for i in self.all_decs.decorators:
            print("-", i.get_version_name())
        if not self.exclude_bbdec:
            print("Note that the BareBonesDecorator is included by default and does not need to be explicitly provided")

        print("\nPlease cite: (no one yet)\n")

    def make_data_cache(self, wrapped_pose: WrappedPose) -> DecoratorDataCache:
        """
        Data caches save time by re-using tensors for nodes and edges you have aleady calculated.
        This usually gives me a 5-10x speedup but your mileage may vary.

        Parameters
        ----------
        wrapped_pose: WrappedPose
            Each pose needs a different cache. Please give us the pose that corresponds to this cache

        Returns
        -------
        cache: DecoratorDataCache
            A data cache that can be passed to generate_input and generate_input_for_resid.
        """

        cache = DecoratorDataCache(wrapped_pose)
        self.all_decs.cache_data(wrapped_pose, cache.dict_cache)
        return cache

    def _calc_nbrs(self, wrapped_pose: WrappedPose, focused_resids: List[int], legal_nbrs: List[int] = None) -> List[int]:
        # includes focus in subset

        if legal_nbrs is None:
            legal_nbrs = wrapped_pose.get_legal_nbrs()  # Still might be None

        focus_xyzs = []
        nbrs = []
        for fres in focused_resids:
            focus_xyzs.append(wrapped_pose.get_atom_xyz(fres, "CA"))
            nbrs.append((-100.0, fres))

        for resid in range(1, wrapped_pose.n_residues() + 1):
            if (resid in focused_resids):
                continue
            if legal_nbrs is not None:
                if not legal_nbrs[resid]:
                    continue
            xyz = wrapped_pose.get_atom_xyz(resid, "CA")
            min_dist = 99999.9
            for fxyz in focus_xyzs:
                min_dist = min(min_dist, np.linalg.norm(xyz - fxyz))
            if min_dist > self.nbr_distance_cutoff_A:
                continue
            nbrs.append((min_dist, resid))

        if len(nbrs) > self.max_residues:
            # print( "AAA", len( nbrs ), self.max_residues )
            nbrs = sorted(nbrs, key=lambda tup: tup[0])
            nbrs = nbrs[0:self.max_residues]
            assert len(nbrs) == self.max_residues

        final_resids = []
        for n in nbrs:
            final_resids.append(n[1])
        return final_resids

    def _get_edge_data_for_pair(self, wrapped_pose: WrappedPose, resid_i: int, resid_j: int, data_cache):
        if data_cache.edge_cache is not None:
            if resid_j in data_cache.edge_cache[resid_i]:
                assert resid_i in data_cache.edge_cache[resid_j]
                return data_cache.edge_cache[resid_i][resid_j], data_cache.edge_cache[resid_j][resid_i]

        f_ij, f_ji = self.all_decs.calc_edge_features(wrapped_pose, resid1=resid_i, resid2=resid_j, dict_cache=data_cache.dict_cache)
        assert len(f_ij) == self.all_decs.n_edge_features()
        assert len(f_ji) == self.all_decs.n_edge_features()

        f_ij = np.asarray(f_ij)
        f_ji = np.asarray(f_ji)
        if data_cache.edge_cache is not None:
            data_cache.edge_cache[resid_i][resid_j] = f_ij
            data_cache.edge_cache[resid_j][resid_i] = f_ji
        return f_ij, f_ji

    def _calc_adjacency_matrix_and_edge_data(self, wrapped_pose: WrappedPose, all_resids: List[int], data_cache):
        N, F, S = self.get_N_F_S()
        A_dense = np.zeros(shape=[N, N])
        E_dense = np.zeros(shape=[N, N, S])

        for i in range(0, len(all_resids) - 1):
            resid_i = all_resids[i]
            i_xyz = wrapped_pose.get_atom_xyz(resid_i, "CA")
            for j in range(i + 1, len(all_resids)):
                resid_j = all_resids[j]
                j_xyz = wrapped_pose.get_atom_xyz(resid_j, "CA")
                dist = np.linalg.norm(i_xyz - j_xyz)
                if dist < self.edge_distance_cutoff_A:
                    f_ij, f_ji = self._get_edge_data_for_pair(wrapped_pose, resid_i=resid_i, resid_j=resid_j, data_cache=data_cache)
                    A_dense[i][j] = 1.0
                    E_dense[i][j] = f_ij

                    A_dense[j][i] = 1.0
                    E_dense[j][i] = f_ji

        return A_dense, E_dense

    def _get_node_data(self, wrapped_pose: WrappedPose, resids: List[int], data_cache):
        N, F, S = self.get_N_F_S()
        X = np.zeros(shape=[N, F])
        index = -1
        for resid in resids:
            index += 1
            if data_cache.node_cache is not None:
                if data_cache.node_cache[resid] is not None:
                    X[index] = data_cache.node_cache[resid]
                    if not self.exclude_bbdec:
                        # Redo focus residues
                        new_bbdec = self.bare_bones_decorator.calc_node_features(wrapped_pose, resid)
                        assert len(new_bbdec) == 1
                        X[index][0] = new_bbdec[0]
                    continue

            n = self.all_decs.calc_node_features(wrapped_pose, resid)

            n = np.asarray(n)
            if data_cache.node_cache is not None:
                data_cache.node_cache[resid] = n
            X[index] = n
        if not self.exclude_bbdec:
            # assumes at least one focus resid
            if X[0][0] != 1:
                print("Error: X[ 0 ][ 0 ] == ", X[0][0])
                for i in range(0, len(resids)):
                    print(i, resids[i], X[i][0])
            assert X[0][0] == 1
        return X

    def generate_XAE_input_tensors(self) -> Tuple[Layer, Layer, Layer]:
        """
        This is just a safe way to create the input layers for your keras model with confidence that they are the right shape

        Returns
        -------
        X_in: Layer
            Node Feature Input
        A_in: Layer
            Adjacency Matrix Input
        E_in: Layer
            Edge Feature Input
        """

        N, F, S = self.get_N_F_S()
        X_in = Input(shape=(N, F), name='X_in')
        A_in = Input(shape=(N, N), sparse=False, name='A_in')
        E_in = Input(shape=(N, N, S), name='E_in')
        return X_in, A_in, E_in

    def generate_input(self, wrapped_pose: WrappedPose, focus_resids: List[int], data_cache: DecoratorDataCache = None,
                       legal_nbrs: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        This is does the actual work of creating a graph and representing it as tensors

        Parameters
        -------
        wrapped_pose: WrappedPose
            Pose to generate data from
        focus_resids: list of ints
            Which resids are the focus residues?
            We use Rosetta conventions here, so the first residue is resid #1,
            second is #2, and so one. No skips.
        data_cache: DecoratorDataCache
            See make_data_cache for details.
            It is very important that this cache was created from this pose
        legal_nbrs: list of ints
            Which resids are allowed to be neighbors? All resids are legal if this is None

        Returns
        -------
        X: ndarray
            Node Features
        A: ndarray
            Adjacency Matrix
        E: ndarray
            Edge Feature
        meta: list of int
            Metadata. At the moment this is just a list of resids in the same order as they are listed in X, A, and E
        """

        if data_cache is None:
            data_cache = NullDecoratorDataCache()

        self.bare_bones_decorator.set_focused_resids(focus_resids)
        all_resids = self._calc_nbrs(wrapped_pose, focus_resids, legal_nbrs=legal_nbrs)
        # n_nodes = len(all_resids)

        # Node Data
        X = self._get_node_data(wrapped_pose, all_resids, data_cache)

        # Adjacency Matrix and Edge Data
        A, E = self._calc_adjacency_matrix_and_edge_data(wrapped_pose, all_resids, data_cache=data_cache)

        return X, A, E, all_resids

    def generate_input_for_resid(self, wrapped_pose: WrappedPose, resid: int, data_cache: DecoratorDataCache = None,
                                 legal_nbrs: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Only have 1 focus resid?
        Then this is sliiiiiiightly cleaner than generate_input().
        It's completely debatable if this is even worthwhile

        Parameters
        -------
        wrapped_pose: WrappedPose
            Pose to generate data from
        focus_resid: inr
            Which resid is the focus residue?
            We use Rosetta conventions here, so the first residue is resid #1,
            second is #2, and so one. No skips.
        data_cache: DecoratorDataCache
            See make_data_cache for details.
            It is very important that this cache was created from this pose
        legal_nbrs: list of ints
            Which resids are allowed to be neighbors? All resids are legal if this is None

        Returns
        -------
        X: ndarray
            Node Features
        A: ndarray
            Adjacency Matrix
        E: ndarray
            Edge Feature
        meta: list of int
            Metadata. At the moment this is just a list of resids in the same order as they are listed in X, A, and E
        """
        return self.generate_input(wrapped_pose, focus_resids=[resid], data_cache=data_cache, legal_nbrs=legal_nbrs)
