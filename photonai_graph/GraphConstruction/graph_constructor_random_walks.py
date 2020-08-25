import numpy as np
from itertools import islice, combinations
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorRandomWalks(GraphConstructor):
    _estimator_type = "transformer"

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
    * `k_distance` [int]:
        the k nearest neighbours value, for the kNN algorithm.
    * `transform_style` [str, default="mean"]:
        generate an adjacency matrix based on the mean matrix like in Ktena et al.: "mean" or per person "individual"
        Or generate a different matrix for every individual: "individual"
    * `number_of_walks` [int, default=10]:
        number of walks to take to sample the random walk matrix
    * `walk_length` [int, default=10]:
        length of the random walk, as the number of steps
    * `window_size` [int, default=5]:
        size of the sliding window from which to sample to coocurrence of two nodes
    * `no_edge_weight` [int, default=1]:
        whether to return an edge weight (0) or not (1)

    Example
    -------
        constructor = GraphConstructorRandomWalks(k_distance=5,
                          transform_style="individual",
                          number_of_walks=25,
                          fisher_transform=1,
                          use_abs=1)
   """

    def __init__(self,
                 k_distance: int = 10,
                 number_of_walks: int = 10,
                 walk_length: int = 10,
                 window_size: int = 5,
                 no_edge_weight: int = 1,
                 feature_axis=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_distance = k_distance
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.no_edge_weight = no_edge_weight
        self.feature_axis = feature_axis

    @staticmethod
    def random_walk(adjacency, walk_length, num_walks):
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
    def sliding_window(seq, n):
        """Returns a sliding window (of width n) over data from the iterable
            s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

        # todo: check return value
        return  # sliding window co-ocurrence

    def sliding_window_frequency(self, X_mean, walk_list):

        coocurrence = np.zeros((X_mean.shape[0], X_mean.shape[1]))

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

    def get_ho_adjacency(self, adj):
        """Returns the higher order adjacency of a graph"""
        adjacency_list = []
        adj = np.squeeze(adj)
        for i in range(adj.shape[0]):
            d, idx = self.distance_sklearn_metrics(adj[i, :, :], k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            # normalize adjacency
            if self.no_edge_weight == 1:
                adjacency[adjacency > 0] = 1

            adjacency = adjacency.toarray()
            higherorder_adjacency = self.__adjacency_to_dense(adjacency)
            # turn adjacency into numpy matrix for concatenation
            adjacency_list.append(higherorder_adjacency)

        adjacency_list = np.asarray(adjacency_list)
        ho_adjacency = adjacency_list[:, :, :, np.newaxis]

        return ho_adjacency

    def transform_test(self, X):
        """Transforms the matrix using random walks"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.get_ho_adjacency(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def transform(self, X):
        # transform each individual or make a mean matrix
        if self.transform_style == "mean":
            # use the mean 2d image of all samples for creating the different photonai_graph structures
            # todo: duplicated code starting here.
            X_mean = np.squeeze(np.mean(X, axis=0))

            # select the proper matrix in case you have multiple
            if np.ndim(X_mean) == 3:
                X_mean = X_mean[:, :, self.adjacency_axis]
            elif np.ndim(X_mean) == 2:
                X_mean = X_mean
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            d, idx = self.distance_sklearn_metrics(X_mean, k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            adjacency = adjacency.toarray()
            if self.no_edge_weight == 1:
                adjacency[adjacency > 0] = 1

            higherorder_adjacency = self.__adjacency_to_dense(adjacency)

            # convert this adjacency matrix to dense format
            higherorder_adjacency = higherorder_adjacency.toarray()

            # reshape X to add the new adjacency
            X_transformed = X[:, :, :, self.feature_axis]
            X_transformed = np.reshape(X_transformed, (X_transformed.shape[0],
                                                       X_transformed.shape[1],
                                                       X_transformed.shape[2],
                                                       -1))
            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency = np.repeat(higherorder_adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
            X_transformed = np.concatenate((adjacency, X_transformed), axis=3)

        elif self.transform_style == "individual":
            X_rw = X.copy()
            # select the proper matrix in case you have multiple
            if np.ndim(X_rw) == 4:
                X_features = X_rw[:, :, :, self.feature_axis]
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
                X_rw = X_rw[:, :, :, self.adjacency_axis]
            elif np.ndim(X_rw) == 3:
                X_rw = X_rw
                X_features = X_rw.copy()
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            adjacency_list = []
            for i in range(X.shape[0]):
                d, idx = self.distance_sklearn_metrics(X_rw[i, :, :], k=self.k_distance, metric='euclidean')
                adjacency = self.adjacency(d, idx).astype(np.float32)

                # normalize adjacency
                if self.no_edge_weight == 1:
                    adjacency[adjacency > 0] = 1

                adjacency = adjacency.toarray()
                higherorder_adjacency = self.__adjacency_to_dense(adjacency)
                # turn adjacency into numpy matrix for concatenation
                adjacency_list.append(higherorder_adjacency)

            adjacency_list = np.asarray(adjacency_list)
            adjacency_list = adjacency_list.reshape((adjacency_list.shape[0], adjacency_list.shape[1],
                                                     adjacency_list.shape[2], -1))
            X_transformed = np.concatenate((adjacency_list, X_features), axis=3)

        else:
            raise KeyError('Only mean and individual transform are supported. '
                           'Please check your spelling for the parameter transform_style.')

        return X_transformed

    # !===== Helper Function =====!
    def __adjacency_to_dense(self, adjacency):
        adjacency_rowsum = np.sum(adjacency, axis=1)
        adjacency_norm = adjacency / adjacency_rowsum[:, np.newaxis]

        walks = self.random_walk(adjacency=adjacency_norm, walk_length=self.walk_length,
                                 num_walks=self.number_of_walks)

        higherorder_adjacency = self.sliding_window_frequency(X_mean=adjacency, walk_list=walks)

        # obtain the kNN photonai_graph from the new adjacency matrix
        d, idx = self.distance_sklearn_metrics(higherorder_adjacency, k=10, metric='euclidean')
        higherorder_adjacency = self.adjacency(d, idx).astype(np.float32)

        # convert this adjacency matrix to dense format
        return higherorder_adjacency.toarray()
