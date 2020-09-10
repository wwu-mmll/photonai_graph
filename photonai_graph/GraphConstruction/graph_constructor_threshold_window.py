from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorThresholdWindow(GraphConstructor):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Thresholds matrix based
    on a chosen threshold window.


    Parameters
    ----------
    * `threshold_upper` [float]:
        upper limit of the threshold window
    * `threshold_lower` [float]:
        lower limit of the threshold window
    * `adjacency_axis` [int]:
        position of the adjacency matrix, default being zero
    * `concatenation_axis` [int]:
        Axis along which to concatenate the adjacency and feature matrix
    * `one_hot_nodes` [int]:
        Whether to generate a one hot encoding of the nodes in the matrix.
    * `return_adjacency_only` [int]:
        whether to return the adjacency matrix only (1) or also a feature matrix (0)
    * `fisher_transform` [int]:
        Perform a fisher transform of each matrix. No (0) or Yes (1)
    * `use_abs` [int]:
        Changes the values to absolute values. Is applied after fisher transform and before
        z-score transformation
    * `zscore` [int, default=0]:
        performs a zscore transformation of the data. Applied after fisher transform and np_abs
        eval_final_perfomance is set to True
    * `use_abs_zscore` [int, default=0]:
        whether to use the absolute values of the z-score transformation or allow for negative
        values. Applied after fisher transform, use_abs and zscore
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default=None]:
        Path to the log data    

    Example
    -------
        constructor = GraphConstructorThresholdWindow(threshold=0.5,
                                                      fisher_transform=1,
                                                      use_abs=1)
   """

    def __init__(self,
                 threshold_upper: float = 1,
                 threshold_lower: float = 0.8,
                 retain_weights: int = 0,
                 transform_style: str = "individual",
                 one_hot_nodes: int = 0,
                 fisher_transform: int = 0,
                 use_abs: int = 0,
                 zscore: int = 0,
                 use_abs_zscore: int = 0,
                 adjacency_axis: int = 0,
                 logs: str = ''):
        super(GraphConstructorThresholdWindow, self).__init__(transform_style=transform_style,
                                                              one_hot_nodes=one_hot_nodes,
                                                              fisher_transform=fisher_transform,
                                                              use_abs=use_abs,
                                                              zscore=zscore,
                                                              use_abs_zscore=use_abs_zscore,
                                                              adjacency_axis=adjacency_axis,
                                                              logs=logs)
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.retain_weights = retain_weights

    def transform(self, X):
        """Transform input matrices accordingly"""
        adj, feat = self.get_mtrx(X)
        # do preparatory matrix transformations
        adj = self.prep_mtrx(adj)
        # threshold matrix
        adj = self.threshold_window(adj)
        # get feature matrix
        X_transformed = self.get_features(adj, feat)

        return X_transformed

    def threshold_window(self, adjacency):
        """Threshold matrix"""
        if self.retain_weights == 0:
            adjacency[adjacency > self.threshold_upper] = 0
            adjacency[(adjacency < self.threshold_upper) & (adjacency >= self.threshold_lower)] = 1
            adjacency[adjacency < self.threshold_lower] = 0
        elif self.retain_weights == 1:
            adjacency[adjacency > self.threshold_upper] = 0
            adjacency[adjacency < self.threshold_lower] = 0
        else:
            raise ValueError('retain weights needs to be 0 or 1')

        return adjacency
