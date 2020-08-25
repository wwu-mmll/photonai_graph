import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorThreshold(GraphConstructor):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Thresholds matrix based
    on a chosen threshold value.


    Parameters
    ----------
    * `threshold` [float]:
        threshold value below which to set matrix entries to zero
    * `retain_weights` [int]:
        whether to retain weight values or not
    * `logs` [str, default=None]:
        Path to the log data    

    Example
    -------
        constructor = GraphConstructorThreshold(threshold=0.5,
                                                fisher_transform=1,
                                                use_abs=1)
   """

    def __init__(self, threshold=0.1,
                 concatenation_axis=3,
                 return_adjacency_only=0,
                 retain_weights=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.concatenation_axis = concatenation_axis
        self.return_adjacency_only = return_adjacency_only
        self.retain_weights = retain_weights

    def transform(self, X):
        """Transform input matrices using a threshold"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.threshold_matrix(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def threshold_matrix(self, adjacency):
        """Threshold matrix"""
        if self.retain_weights == 0:
            adjacency[adjacency > self.threshold] = 1
            adjacency[adjacency < self.threshold] = 0
        elif self.retain_weights == 1:
            adjacency[adjacency < self.threshold] = 0
        else:
            raise ValueError('retain weights needs to be 0 or 1')

        return adjacency
