import os
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorPercentageWindow(GraphConstructor):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the top x percent
    of connections and sets all other connections to zero


    Parameters
    ----------
    * `percentage_upper` [float]:
        upper limit of the percentage window
    * `percentage_lower` [float]:
        lower limit of the percentage window
        
    Example
    -------
        constructor = GraphConstructorPercentageWindow(percentage=0.9,
                                                       fisher_transform=1,
                                                       use_abs=1)
   """

    def __init__(self,
                 percentage_upper: float = 50,
                 percentage_lower: float = 10,
                 retain_weights: int = 0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.percentage_upper = percentage_upper
        self.percentage_lower = percentage_lower
        self.retain_weights = retain_weights

    def transform_test(self, X):
        """Select percent strongest connections"""
        adj, feat = self.get_mtrx(X)
        # prepare matrices
        adj = self.prep_mtrx(adj)
        # get percent strongest connections
        adj = self.percentage_window(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def percentage_window(self, adjacency):
        """Finds the x percent strongest connections"""
        for matrix in range(adjacency.shape[0]):
            thresh_upper = np.percentile(adjacency[matrix, :, :, :], self.percentage_upper)
            thresh_lower = np.percentile(adjacency[matrix, :, :, :], self.percentage_lower)
            if self.retain_weights == 0:
                adjacency[matrix, :, :, :][adjacency[matrix, :, :, :] >= thresh_upper] = 0
                adjacency[matrix, :, :, :][(adjacency[matrix, :, :, :] <= thresh_upper) & (adjacency[matrix, :, :, :] >= thresh_lower)] = 1
                adjacency[matrix, :, :, :][adjacency[matrix, :, :, :] <= thresh_lower] = 0
            elif self.retain_weights == 1:
                adjacency[adjacency[matrix, :, :, :][matrix, :, :, :] >= thresh_upper] = 0
                adjacency[adjacency[matrix, :, :, :][matrix, :, :, :] <= thresh_lower] = 0
            else:
                raise ValueError('retain weights needs to be 0 or 1')

        return adjacency
