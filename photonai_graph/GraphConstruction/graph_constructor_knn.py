import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorKNN(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 k_distance: int = 10,
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
        from connectivity matrices. Selects the k nearest
        neighbours for each node based on pairwise distance.
        Recommended for functional connectivity.
        Adapted from Ktena et al, 2017.


        Parameters
        ----------
        k_distance: int
            the k nearest neighbours value, for the kNN algorithm.
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
        constructor = GraphConstructorKNN(k_distance=6,
                                          fisher_transform=1,
                                          use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorKNN',
                                    hyperparameters={'k_distance': 6}))
        ```
       """
        super(GraphConstructorKNN, self).__init__(one_hot_nodes=one_hot_nodes,
                                                  use_abs=use_abs,
                                                  fisher_transform=fisher_transform,
                                                  use_abs_fisher=use_abs_fisher,
                                                  zscore=zscore,
                                                  use_abs_zscore=use_abs_zscore,
                                                  adjacency_axis=adjacency_axis,
                                                  logs=logs)
        self.k_distance = k_distance

    def get_knn(self, adjacency: np.ndarray) -> np.ndarray:
        """Returns kNN matrices"""
        adjacency = np.squeeze(adjacency)
        adjacency_list = []
        for i in range(adjacency.shape[0]):
            # generate adjacency matrix
            d, idx = self.distance_sklearn_metrics(adjacency[i, :, :], k=self.k_distance, metric='euclidean')
            k_adjacency = self.adjacency(d, idx).astype(np.float32)

            # turn adjacency into numpy matrix for concatenation
            k_adjacency = k_adjacency.toarray()
            adjacency_list.append(k_adjacency)

        # X = X[..., None] + adjacency[None, None, :] #use broadcasting to speed up computation
        adjacency_knn = np.asarray(adjacency_list)
        adjacency_knn = adjacency_knn[:, :, :, np.newaxis]

        return adjacency_knn

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform matrices based on k nearest neighbours"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.get_knn(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed
