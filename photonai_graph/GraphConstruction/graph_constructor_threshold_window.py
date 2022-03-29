import numpy as np

from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorThresholdWindow(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 threshold_upper: float = 1,
                 threshold_lower: float = 0.8,
                 retain_weights: float = 0,
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
        on a chosen threshold window.


        Parameters
        ----------
        threshold_upper: float
            upper limit of the threshold window
        threshold_lower: float
            lower limit of the threshold window
        retain_weights: int
            whether to retain weight values or not
        one_hot_nodes: int
            Whether to generate a one hot encoding of the nodes in the matrix.
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int
            Perform a fisher transform of each matrix. No (0) or Yes (1)
        use_abs_fisher: int
            Changes the values to absolute values. Is applied after fisher transform and before
            z-score transformation
        zscore: int, default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
            eval_final_perfomance is set to True
        use_abs_zscore: int, default=0
            whether to use the absolute values of the z-score transformation or allow for negative
            values. Applied after fisher transform, use_abs and zscore
        adjacency_axis: int
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data

        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorThresholdWindow(threshold_upper=0.7,
                                                      threshold_lower=0.3,
                                                      use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorThresholdWindow',
                                    hyperparameters={'threshold_upper': 0.7, 'threshold_lower': 0.3}))
        ```
       """
        super(GraphConstructorThresholdWindow, self).__init__(one_hot_nodes=one_hot_nodes,
                                                              use_abs=use_abs,
                                                              fisher_transform=fisher_transform,
                                                              use_abs_fisher=use_abs_fisher,
                                                              zscore=zscore,
                                                              use_abs_zscore=use_abs_zscore,
                                                              adjacency_axis=adjacency_axis,
                                                              logs=logs)
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.retain_weights = retain_weights
        if retain_weights not in [0, 1]:
            raise ValueError("retain_weights has to be in [0, 1]")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input matrices accordingly"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.threshold_window(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def threshold_window(self, adjacency: np.ndarray) -> np.ndarray:
        """Threshold matrix"""
        adjacency[adjacency < self.threshold_lower] = 0
        adjacency[adjacency > self.threshold_upper] = 0
        if not self.retain_weights:
            adjacency[(adjacency < self.threshold_upper) & (adjacency >= self.threshold_lower)] = 1
        return adjacency
