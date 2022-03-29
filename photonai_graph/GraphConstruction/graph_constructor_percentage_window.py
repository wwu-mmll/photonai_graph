import numpy as np
from photonai_graph.GraphConstruction.graph_constructor import GraphConstructor


class GraphConstructorPercentageWindow(GraphConstructor):
    _estimator_type = "transformer"

    def __init__(self,
                 percentage_upper: float = 50,
                 percentage_lower: float = 10,
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
        percentage_upper: float
            upper limit of the percentage window
        percentage_lower: float
            lower limit of the percentage window
        one_hot_nodes: int
            Whether to generate a one hot encoding of the nodes in the matrix (1) or not (0)
        use_abs: bool, default = False
            whether to convert all matrix values to absolute values before applying
            other transformations
        fisher_transform: int
            whether to perform a fisher transform of each matrix (1) or not (0)
        use_abs_fisher: int,default=0
            changes the values to absolute values. Is applied after fisher transform and before z-score transformation
        zscore: int,default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
        use_abs_zscore: int,default=0
            whether to use the absolute values of the z-score transformation or allow for negative values
        adjacency_axis: int
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data

        Example
        -------
        Use outside of a PHOTON pipeline

        ```python
        constructor = GraphConstructorPercentageWindow(percentage_upper=0.9,
                                                       percentage_lower=0.7
                                                       fisher_transform=1,
                                                       use_abs=1)
        ```

        Or as part of a pipeline

        ```python
        my_pipe.add(PipelineElement('GraphConstructorPercentageWindow',
                                    hyperparameters={'percentage_upper': 0.9, 'percentage_lower': 0.7}))
        ```
       """
        super(GraphConstructorPercentageWindow, self).__init__(one_hot_nodes=one_hot_nodes,
                                                               use_abs=use_abs,
                                                               fisher_transform=fisher_transform,
                                                               use_abs_fisher=use_abs_fisher,
                                                               zscore=zscore,
                                                               use_abs_zscore=use_abs_zscore,
                                                               adjacency_axis=adjacency_axis,
                                                               logs=logs)
        self.percentage_upper = percentage_upper
        self.percentage_lower = percentage_lower
        self.retain_weights = retain_weights
        if retain_weights not in [1, 0]:
            raise ValueError('retain_weights has to be in [1, 0]')

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select percent strongest connections"""
        adj, feat = self.get_mtrx(X)
        # prepare matrices
        adj = self.prep_mtrx(adj)
        # get percent strongest connections
        adj = self.percentage_window(adj)
        # get feature matrix
        x_transformed = self.get_features(adj, feat)

        return x_transformed

    def percentage_window(self, adjacency: np.ndarray) -> np.ndarray:
        """Finds the x percent strongest connections"""
        for matrix in range(adjacency.shape[0]):
            thresh_upper = np.percentile(adjacency[matrix, :, :, :], self.percentage_upper)
            thresh_lower = np.percentile(adjacency[matrix, :, :, :], self.percentage_lower)
            adjacency[matrix, :, :, :][adjacency[matrix, :, :, :] <= thresh_lower] = 0
            adjacency[matrix, :, :, :][adjacency[matrix, :, :, :] >= thresh_upper] = 0
            if not self.retain_weights:
                adjacency[matrix, :, :, :][(adjacency[matrix, :, :, :] <= thresh_upper) &
                                           (adjacency[matrix, :, :, :] >= thresh_lower)] = 1

        return adjacency
