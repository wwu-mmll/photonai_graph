import os
from abc import ABC
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from scipy import sparse
from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform


class GraphConstructor(BaseEstimator, TransformerMixin, ABC):
    _estimator_type = "transformer"

    def __init__(self,
                 one_hot_nodes: int = 0,
                 fisher_transform: int = 0,
                 discard_original_connectivity: bool = False,
                 use_abs: int = 0,
                 zscore: int = 0,
                 use_abs_zscore: int = 0,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
        Base class for all graph constructors. Implements
        methods shared by different constructors.

        Parameters
        ----------
        one_hot_nodes: int, default=0
            whether to return a one hot node encoding as feature or not
        fisher_transform: int, default=0
            whether to perform a fisher transform of every matrix
        discard_original_connectivity: bool,default=False
            If true the second index of the last dimension will be the original connectivity.
            Otherwise the original connectivity will be dropped from the matrix.
        use_abs: int
            Changes the values to absolute values. Is applied after fisher transform and before
            z-score transformation
        zscore: int, default=0
            performs a zscore transformation of the data. Applied after fisher transform and np_abs
            eval_final_perfomance is set to True
        use_abs_zscore: int, default=0
            whether to use the absolute values of the z-score transformation or allow for negative
            values. Applied after fisher transform, use_abs and zscore
        adjacency_axis: int, default=0
            position of the adjacency matrix, default being zero
        logs: str, default=None
            Path to the log data
        """
        if one_hot_nodes not in [0, 1]:
            raise ValueError("one_hot_nodes must be in [0, 1]")
        self.one_hot_nodes = one_hot_nodes
        self.fisher_transform = fisher_transform
        self.discard_original_connectivity = discard_original_connectivity
        self.use_abs = use_abs
        self.zscore = zscore
        self.use_abs_zscore = use_abs_zscore
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        self.mean_matrix = None

    def fit(self, X, y):
        return self

    def get_mtrx(self, X):
        """Returns a feature and adjacency matrix"""

        # ensure that the array has the "right" number of dimensions
        if np.ndim(X) == 4:
            adjacency_matrix = X[:, :, :, self.adjacency_axis].copy()
            adjacency_matrix = adjacency_matrix[:, :, :, np.newaxis]  # return 4d
            feature_matrix = X.copy()
        # handle the case where there are 3 dimensions, meaning that there is no "adjacency axis"
        elif np.ndim(X) == 3:
            adjacency_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            feature_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
        else:
            raise ValueError('input matrix needs to have 3 or 4 dimensions')

        return adjacency_matrix, feature_matrix

    def prep_mtrx(self, adjacency):
        """transforms the matrix according to selected criteria"""
        if self.fisher_transform:
            adjacency = individual_fishertransform(adjacency)
        if self.use_abs:
            adjacency = np.abs(adjacency)
        if self.zscore:
            adjacency = individual_ztransform(adjacency)
        if self.use_abs_zscore:
            adjacency = np.abs(adjacency)

        return adjacency

    def get_features(self, adjacency, features):
        """Returns a concatenated version of feature matrix and one hot nodes"""
        if self.discard_original_connectivity:
            features = features[..., 1:]
        if self.one_hot_nodes:
            adjacency = self.get_one_hot_nodes(adjacency)
        transformed = np.concatenate((adjacency, features), axis=3)
        return transformed

    @staticmethod
    def get_one_hot_nodes(X):
        """Returns a one hot node encoding for every node of the matrix"""
        # an identity matrix is a one hot node encoding
        identity_matrix = np.identity((X.shape[1]))
        # expands dims
        identity_matrix = identity_matrix[np.newaxis, :, :, np.newaxis]
        # repeat along first dimension and concatenate
        one_hot = np.repeat(identity_matrix, X.shape[0], 0)
        concat = np.concatenate((X, one_hot), axis=3)
        return concat

    @staticmethod
    def adjacency(dist, idx):
        """Return the adjacency matrix of a kNN photonai_graph."""
        m, k = dist.shape
        assert m, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        i = np.arange(0, m).repeat(k)
        j = idx.reshape(m * k)
        v = dist.reshape(m * k)
        w = sparse.coo_matrix((v, (i, j)), shape=(m, m))

        # No self-connections.
        w.setdiag(0)

        # Non-directed photonai_graph.
        bigger = w.T > w
        w = w - w.multiply(bigger) + w.T.multiply(bigger)

        assert w.nnz % 2 == 0
        assert np.abs(w - w.T).mean() < 1e-10
        assert type(w) is sparse.csr.csr_matrix

        return w

    @staticmethod
    def distance_sklearn_metrics(z, k, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = pairwise_distances(z, metric=metric, n_jobs=-2)
        # k-NN photonai_graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx
