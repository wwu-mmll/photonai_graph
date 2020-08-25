import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorPercentage(GraphConstructor):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the top x percent
    of connections and sets all other connections to zero


    Parameters
    ----------
    * `percentage` [float]:
        value of percent of connections to discard. A value of 0.9 keeps only the top 10%
    * `retain_weights` [int]:
        whether to retain weight values or not

    Example
    -------
        constructor = GraphConstructorPercentage(percentage=0.9,
                                                 fisher_transform=1,
                                                 use_abs=1)
   """

    def __init__(self, percentage: float = 0.8,
                 retain_weights: int = 0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.percentage = percentage
        self.retain_weights = retain_weights

    def transform(self, X):
        """Select percent strongest connections"""
        adj, feat = self.get_mtrx(X)
        # prepare matrices
        adj = self.prep_mtrx(adj)
        # get percent strongest connections
        adj = self.percent_strongest(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def percent_strongest(self, X):
        """Finds the x percent strongest connections"""
        for matrix in range(X.shape[0]):
            thresh = np.percentile(X[matrix, :, :, :], self.percentage)
            if self.retain_weights == 0:
                X[matrix, :, :, :][X[matrix, :, :, :] > thresh] = 1
                X[matrix, :, :, :][X[matrix, :, :, :] <= thresh] = 0
            elif self.retain_weights == 1:
                X[matrix, :, :, :][X[matrix, :, :, :] <= thresh] = 0
            else:
                raise ValueError('retain weights needs to be 0 or 1')

        return X
