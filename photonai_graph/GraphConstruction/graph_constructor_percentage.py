import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorPercentage(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 percentage: float = 80,
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
        from connectivity matrices. Selects the top x percent
        of connections and sets all other connections to zero


        Parameters
        ----------
        percentage: float
            value of percent of connections to discard. A value of 90 keeps only the top 10%
        retain_weights: float
            whether to retain weight values or not
        one_hot_nodes: int, default=0
            Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0)
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int, default=0
            whether to perform a fisher transform of each matrix (1) or not (0)
        use_abs_fisher: int, default=0
            changes the values to absolute values. Is applied after fisher transform and before z-score transformation
        zscore: int, default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
        use_abs_zscore: int, default=0
            whether to use the absolute values of the z-score transformation or allow for negative values
        adjacency_axis: int, default=0
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data

        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorPercentage(percentage=0.9,
                                                 fisher_transform=1,
                                                 use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorPercentage',
                                    hyperparameters={'percentage': 0.9}))
        ```


       """
        super(GraphConstructorPercentage, self).__init__(one_hot_nodes=one_hot_nodes,
                                                         use_abs=use_abs,
                                                         fisher_transform=fisher_transform,
                                                         use_abs_fisher=use_abs_fisher,
                                                         zscore=zscore,
                                                         use_abs_zscore=use_abs_zscore,
                                                         adjacency_axis=adjacency_axis,
                                                         logs=logs)
        self.percentage = percentage
        self.retain_weights = retain_weights
        if retain_weights not in [0, 1]:
            raise ValueError("retain_weights has to be in [0, 1]")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select percent strongest connections"""
        adj, feat = self.get_mtrx(X)
        # prepare matrices
        adj = self.prep_mtrx(adj)
        # get percent strongest connections
        adj = self.percent_strongest(adj)
        # get feature matrix
        x_transformed = self.get_features(adj, feat)

        return x_transformed

    def percent_strongest(self, adjacency: np.ndarray) -> np.ndarray:
        """Finds the x percent strongest connections"""
        for matrix in range(adjacency.shape[0]):
            thresh = np.percentile(adjacency[matrix, ...], self.percentage)
            if np.isnan(thresh):
                raise ValueError('Input contains NaN -> can not threshold')
            else:
                adjacency[matrix, ...][adjacency[matrix, ...] < thresh] = 0
                if not self.retain_weights:
                    adjacency[matrix, ...][adjacency[matrix, ...] >= thresh] = 1

        return adjacency
