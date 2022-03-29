from typing import List
from itertools import islice, combinations

import numpy as np

from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorRandomWalks(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 k_distance: int = 10,
                 number_of_walks: int = 10,
                 walk_length: int = 10,
                 window_size: int = 5,
                 no_edge_weight: int = 1,
                 one_hot_nodes: int = 0,
                 use_abs: int = 0,
                 fisher_transform: int = 0,
                 use_abs_fisher: int = 0,
                 zscore: int = 0,
                 use_abs_zscore: int = 0,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
        Transformer class for generating adjacency matrices
        from connectivity matrices. Generates a kNN matrix
        and performs random walks on these. The coocurrence
        of two nodes in those walks is then used to generate
        a higher-order adjacency matrix, by applying the kNN
        algorithm on the matrix again.
        Adapted from Ma et al., 2019.


        Parameters
        ----------
        k_distance: int
            the k nearest neighbours value, for the kNN algorithm.
        number_of_walks: int,default=10
            number of walks to take to sample the random walk matrix
        walk_length: int,default=10
            length of the random walk, as the number of steps
        window_size: int,default=5
            size of the sliding window from which to sample to coocurrence of two nodes
        no_edge_weight: int,default=1
            whether to return an edge weight (0) or not (1)
        one_hot_nodes: int,default=0
            Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0)
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int,default=0
            whether to perform a fisher transform of each matrix (1) or not (0)
        use_abs_fisher: int,default=0
            changes the values to absolute values. Is applied after fisher transform and before z-score transformation
        zscore: int,default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
        use_abs_zscore: int,default=0
            whether to use the absolute values of the z-score transformation or allow for negative values
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data


        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorRandomWalks(k_distance=5,
                                                  number_of_walks=25,
                                                  fisher_transform=1,
                                                  use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorRandomWalks',
                                    hyperparameters={'k_distance': 5,
                                    'number_of_walks': 25}))
        ```
       """
        super(GraphConstructorRandomWalks, self).__init__(one_hot_nodes=one_hot_nodes,
                                                          use_abs=use_abs,
                                                          fisher_transform=fisher_transform,
                                                          use_abs_fisher=use_abs_fisher,
                                                          zscore=zscore,
                                                          use_abs_zscore=use_abs_zscore,
                                                          adjacency_axis=adjacency_axis,
                                                          logs=logs)
        self.k_distance = k_distance
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.no_edge_weight = no_edge_weight

    @staticmethod
    def random_walk(adjacency: np.ndarray, walk_length: int, num_walks: int) -> List[List[List[int]]]:
        """Performs a random walk on a given photonai_graph"""
        # a -> adj
        # i -> starting row
        walks = []  # holds transitions
        elements = np.arange(adjacency.shape[0])  # for our photonai_graph [0,1,2,3]
        for k in range(num_walks):
            node_walks = []
            for i in range(elements.shape[0]):
                index = np.random.choice(elements, replace=False)  # current index for this iteration
                count = 0  # count of transitions
                walk = []
                while count < walk_length:
                    count += 1
                    walk.append(index)
                    probs = adjacency[index]  # probability of transitions
                    # sample from probs
                    sample = np.random.choice(elements, p=probs)  # sample a target using probs
                    index = sample  # go to target
                node_walks.append(walk)
            walks.append(node_walks)

        return walks

    @staticmethod
    def sliding_window(seq, n: int):
        """Returns a sliding window (of width n) over data from the iterable
            s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

        # todo: check return value!
        return  # sliding window co-ocurrence

    def sliding_window_frequency(self, x_mean: np.ndarray, walk_list: List[List[List[int]]]) -> np.ndarray:

        coocurrence = np.zeros((x_mean.shape[0], x_mean.shape[1]))

        for i in walk_list:
            node_walks = i
            for x in node_walks:
                for w in self.sliding_window(x, self.window_size):
                    for subset in combinations(w, 2):
                        coord1 = subset[0]
                        coord2 = subset[1]
                        coocurrence[coord1, coord2] += 1
                    # if they occur at the same time add + 1 to the frequency table

        return coocurrence

    def get_ho_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """Returns the higher order adjacency of a graph"""
        adj = np.squeeze(adj)

        adjacency_list = self.__adjacency_to_list(adj, adj)
        ho_adjacency = adjacency_list[..., np.newaxis]

        return ho_adjacency

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the matrix using random walks"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.get_ho_adjacency(adj)
        # get feature matrix
        x_transformed = self.get_features(adj, feat)

        return x_transformed

    # !===== Helper Functions =====!
    def __adjacency_to_list(self, adjacency_in: np.ndarray, adjacency_rw: np.ndarray) -> np.ndarray:
        adjacency_list = []
        for i in range(adjacency_in.shape[0]):
            d, idx = self.distance_sklearn_metrics(adjacency_rw[i, :, :], k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            # normalize adjacency
            if self.no_edge_weight == 1:
                adjacency[adjacency > 0] = 1

            adjacency = adjacency.toarray()
            higherorder_adjacency = self.__adjacency_to_dense(adjacency)
            # turn adjacency into numpy matrix for concatenation
            adjacency_list.append(higherorder_adjacency)

        return np.asarray(adjacency_list)

    def __adjacency_to_dense(self, adjacency: np.ndarray) -> np.ndarray:
        adjacency_rowsum = np.sum(adjacency, axis=1)
        adjacency_norm = adjacency / adjacency_rowsum[:, np.newaxis]

        walks = self.random_walk(adjacency=adjacency_norm, walk_length=self.walk_length,
                                 num_walks=self.number_of_walks)

        higherorder_adjacency = self.sliding_window_frequency(x_mean=adjacency, walk_list=walks)

        # obtain the kNN photonai_graph from the new adjacency matrix
        d, idx = self.distance_sklearn_metrics(higherorder_adjacency, k=10, metric='euclidean')
        higherorder_adjacency = self.adjacency(d, idx).astype(np.float32)

        # convert this adjacency matrix to dense format
        return higherorder_adjacency.toarray()
