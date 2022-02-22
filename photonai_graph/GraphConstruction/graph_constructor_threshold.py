from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorThreshold(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self, threshold: float = 0.1,
                 concatenation_axis: int = 3,
                 return_adjacency_only: int = 0,
                 retain_weights: int = 0,
                 transform_style: str = "individual",
                 one_hot_nodes: int = 0,
                 fisher_transform: int = 0,
                 use_abs: int = 0,
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
        threshold: float:
            threshold value below which to set matrix entries to zero
        retain_weights: int
            whether to retain weight values or not
        logs: str, default=None
            Path to the log data

        Example
        -------
            constructor = GraphConstructorThreshold(threshold=0.5,
                                                    fisher_transform=1,
                                                    use_abs=1)
       """
        super(GraphConstructorThreshold, self).__init__(transform_style=transform_style,
                                                        one_hot_nodes=one_hot_nodes,
                                                        fisher_transform=fisher_transform,
                                                        use_abs=use_abs,
                                                        zscore=zscore,
                                                        use_abs_zscore=use_abs_zscore,
                                                        adjacency_axis=adjacency_axis,
                                                        logs=logs)
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
            adjacency[adjacency >= self.threshold] = 1
            adjacency[adjacency < self.threshold] = 0
        elif self.retain_weights == 1:
            adjacency[adjacency < self.threshold] = 0
        else:
            raise ValueError('retain weights needs to be 0 or 1')

        return adjacency
