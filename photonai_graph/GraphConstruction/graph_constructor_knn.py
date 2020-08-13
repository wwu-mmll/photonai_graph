import os
import sklearn
import numpy as np

from photonai_graph.GraphConstruction.abc_graph_constructor_adjacency import GraphConstructorAdjacency


class GraphConstructorKNN(GraphConstructorAdjacency):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the k nearest
    neighbours for each node based on pairwise distance.
    Recommended for functional connectivity.
    Adapted from Ktena et al, 2017.


    Parameters
    ----------
    * `k_distance` [int]:
        the k nearest neighbours value, for the kNN algorithm.
    * `transform_style` [str, default="mean"]:
        generate an adjacency matrix based on the mean matrix like in Ktena et al.: "mean" 
        Or generate a different matrix for every individual: "individual"
    * `adjacency_axis` [int]:
        position of the adjacency matrix, default being zero
    * `one_hot_nodes` [int]:
        Whether to generate a one hot encoding of the nodes in the matrix.
    * `return_adjacency_only` [int]:
        whether to return the adjacency matrix only (1) or also a feature matrix (0)
    * `fisher_transform` [int]:
        Perform a fisher transform of each matrix. No (0) or Yes (1)
    * `use_abs` [int]:
        Changes the values to absolute values. Is applied after fisher transform and before
        z-score transformation
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default='']:
        Path to the log data    

    Example
    -------
        constructor = GraphConstructorKNN(k_distance=6,
                                          fisher_transform=1,
                                          use_abs=1)
   """

    def __init__(self, k_distance=10, transform_style="mean", adjacency_axis=0, logs=''):
        self.k_distance = k_distance
        self.transform_style = transform_style
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        # todo: is this function really necessary?
        pass

    def distance_sklearn_metrics(self, z, k, metric='euclidean'):
        # todo: check if this function could be static
        """Compute exact pairwise distances."""
        d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx

    def transform(self, X):
        # transform each photonai_graph through its own adjacency or all graphs
        if self.transform_style == "mean" or self.transform_style == "Mean":
            # use the mean 2d image of all samples for creating the different photonai_graph structures
            X_mean = np.squeeze(np.mean(X, axis=0))

            # select the proper matrix in case you have multiple
            if np.ndim(X_mean) == 3:
                X_mean = X_mean[:, :, self.adjacency_axis]
            elif np.ndim(X_mean) == 2:
                X_mean = X_mean
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            # generate adjacency matrix
            d, idx = self.distance_sklearn_metrics(X_mean, k=self.k_distance, metric='euclidean')
            adjacency = self.adjacency(d, idx).astype(np.float32)

            # turn adjacency into numpy matrix for concatenation
            adjacency = adjacency.toarray()

            X_transformed = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], -1))
            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency = np.repeat(adjacency[np.newaxis, :, :, np.newaxis], X.shape[0], axis=0)
            X_transformed = np.concatenate((adjacency, X_transformed), axis=3)

        elif self.transform_style == "individual" or self.transform_style == "Individual":
            # make a copy of x
            X_knn = X.copy()
            # select the proper matrix in case you have multiple
            if np.ndim(X_knn) == 4:
                X_knn = X_knn[:, :, :, self.adjacency_axis]
                X_features = X_knn.copy()
            elif np.ndim(X_knn) == 3:
                X_knn = X_knn
                X_features = X_knn.copy()
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')

            # make a list for all adjacency matrices
            adjacency_list = []
            # generate adjacency matrix for each individual
            for i in range(X_knn.shape[0]):
                d, idx = self.distance_sklearn_metrics(X_knn[i, :, :], k=self.k_distance, metric='euclidean')
                adjacency = self.adjacency(d, idx).astype(np.float32)

                # turn adjacency into numpy matrix for concatenation
                adjacency = adjacency.toarray()
                adjacency_list.append(adjacency)

            # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
            adjacency_list = np.asarray(adjacency_list)
            adjacency_list = adjacency_list.reshape((adjacency_list.shape[0],
                                                     adjacency_list.shape[1],
                                                     adjacency_list.shape[2],
                                                     -1))
            if np.ndim(X_features) == 3:
                X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], X_features.shape[2], -1))
            X_transformed = np.concatenate((adjacency_list, X_features), axis=3)

        else:
            raise KeyError('Only mean and individual transform are supported. '
                           'Please check your spelling for the parameter transform_style.')

        return X_transformed
