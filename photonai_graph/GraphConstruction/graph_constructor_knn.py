import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorKNN(GraphConstructor):
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

    Example
    -------
        constructor = GraphConstructorKNN(k_distance=6,
                                          fisher_transform=1,
                                          use_abs=1)
   """

    def __init__(self,
                 k_distance: int = 10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.k_distance = k_distance

    def get_knn(self, adjacency):
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
        adjacency_kNN = np.asarray(adjacency_list)
        adjacency_kNN = adjacency_kNN[:, :, :, np.newaxis]

        return adjacency_kNN

    def transform(self, X):
        """Transform matrices based on k nearest neighbours"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.get_knn(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed