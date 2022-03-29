import numpy as np

from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorThreshold(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 threshold: float = 0.1,
                 concatenation_axis: int = 3,
                 return_adjacency_only: int = 0,
                 retain_weights: int = 0,
                 discard_original_connectivity: bool = False,
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
        from connectivity matrices. Thresholds matrix based
        on a chosen threshold value.


        Parameters
        ----------
        threshold: float
            threshold value below which to set matrix entries to zero
        concatenation_axis: int
            axis along which to concatenate the adjacency and feature matrix
        return_adjacency_only: int
            whether to return the adjacency matrix only (1) or also a feature matrix (0)
        retain_weights: int
            whether to retain weight values or not
        discard_original_connectivity: bool,default=False
            If true the original connectivity will be passed on. Otherwise it gets discarded
        one_hot_nodes: int
            Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0)
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int
            whether to perform a fisher transform of each matrix (1) or not (0)
        use_abs_fisher: int, default=0
            changes the values to absolute values. Is applied after fisher transform and before z-score transformation
        zscore: int, default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
        use_abs_zscore:
            whether to use the absolute values of the z-score transformation or allow for negative values
        adjacency_axis: int, default=0
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data


        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorThreshold(threshold=0.5,
                                                fisher_transform=1,
                                                use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorThreshold',
                                    hyperparameters={'threshold': 0.5}))
        ```
       """
        super(GraphConstructorThreshold, self).__init__(one_hot_nodes=one_hot_nodes,
                                                        use_abs=use_abs,
                                                        fisher_transform=fisher_transform,
                                                        discard_original_connectivity=discard_original_connectivity,
                                                        use_abs_fisher=use_abs_fisher,
                                                        zscore=zscore,
                                                        use_abs_zscore=use_abs_zscore,
                                                        adjacency_axis=adjacency_axis,
                                                        logs=logs)
        if retain_weights not in [0, 1]:
            raise ValueError("retain_weights has to be in [0, 1]")
        self.threshold = threshold
        self.concatenation_axis = concatenation_axis
        self.return_adjacency_only = return_adjacency_only
        self.retain_weights = retain_weights

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input matrices using a threshold"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.threshold_matrix(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def threshold_matrix(self, adjacency: np.ndarray) -> np.ndarray:
        """Threshold matrix"""
        adjacency[adjacency < self.threshold] = 0
        if not self.retain_weights:
            adjacency[adjacency >= self.threshold] = 1

        return adjacency
