import os
from abc import ABC
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from scipy import sparse
from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform


class GraphConstructor(BaseEstimator, TransformerMixin, ABC):
    _estimator_type = "transformer"

    """
    Base class for all graph constructors. Implements
    methods shared by different constructors.

    Parameters
    ----------
    * `transform_style` [str, default="mean"]
    * `one_hot_nodes` [int, default=0]
        whether to return a one hot node encoding as feature or not
    * `fisher_transform` [int, default=0]:
        whether to perform a fisher transform of every matrix
    * `use_abs` [int]:
        Changes the values to absolute values. Is applied after fisher transform and before
        z-score transformation
    * `zscore` [int, default=0]:
        performs a zscore transformation of the data. Applied after fisher transform and np_abs
        eval_final_perfomance is set to True
    * `use_abs_zscore` [int, default=0]:
        whether to use the absolute values of the z-score transformation or allow for negative
        values. Applied after fisher transform, use_abs and zscore
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    """

    def __init__(self,
                 transform_style: str = "mean",
                 one_hot_nodes: int = 0,
                 fisher_transform: int = 0,
                 use_abs: int = 0,
                 zscore: int = 0,
                 use_abs_zscore: int = 0,
                 adjacency_axis=0,
                 logs: str = ''):
        self.transform_style = transform_style
        self.one_hot_nodes = one_hot_nodes
        self.fisher_transform = fisher_transform
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
        """implements a mean matrix construction"""

        if self.transform_style == "mean" or self.transform_style == "Mean":
            # generate mean matrix
            X_mean = np.squeeze(np.mean(X, axis=0))
            # select the proper matrix in case you have multiple
            if np.ndim(X_mean) == 3:
                X_mean = X_mean[:, :, self.adjacency_axis]
            elif np.ndim(X_mean) == 2:
                X_mean = X_mean
            else:
                raise ValueError('The input matrices need to have 3 or 4 dimensions. Please check your input matrix.')
            # assign mean matrix for later use
            self.mean_matrix = X_mean

        elif self.transform_style == "individual" or self.transform_style == "Individual":
            pass

        else:
            raise ValueError('transform_style needs to be individual or mean')

        return self

    def get_mtrx(self, X):
        """Returns a feature and adjacency matrix"""

        if self.transform_style == "individual":

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
                raise Exception('input matrix needs to have 3 or 4 dimensions')

        elif self.transform_style == "mean":
            adjacency_matrix = self.mean_matrix.copy()
            adjacency_matrix = adjacency_matrix[np.newaxis, :, :, np.newaxis]
            adjacency_matrix = np.repeat(adjacency_matrix, X.shape[0], axis=0)
            if np.ndim(X) == 4:
                feature_matrix = X.copy()
            elif np.ndim(X) == 3:
                feature_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            else:
                raise Exception('input matrix needs to have 3 or 4 dimensions')

        else:
            raise ValueError('transform_style needs to be individual or mean')

        return adjacency_matrix, feature_matrix

    def prep_mtrx(self, adjacency):
        """transforms the matrix according to selected criteria"""
        if self.fisher_transform == 1:
            adjacency = individual_fishertransform(adjacency)
        if self.use_abs == 1:
            adjacency = np.abs(adjacency)
        if self.zscore == 1:
            adjacency = individual_ztransform(adjacency)
        if self.use_abs_zscore == 1:
            adjacency = np.abs(adjacency)

        return adjacency

    def get_features(self, adjacency, features):
        """Returns a concatenated version of feature matrix and one hot nodes"""
        if self.one_hot_nodes == 0:
            X_transformed = np.concatenate((adjacency, features), axis=3)
        elif self.one_hot_nodes == 1:
            X_transformed = self.get_one_hot_nodes(adjacency)
        else:
            raise ValueError('one_hot_nodes needs to be 0 or 1.')

        return X_transformed

    @staticmethod
    def get_one_hot_nodes(X):
        """Returns a one hot node encoding for every node of the matrix"""
        # an identity matrix is a one hot node encoding
        identity_matrix = np.identity((X.shape[1]))
        # expands dims
        identity_matrix = identity_matrix[np.newaxis, :, :, np.newaxis]
        # repeat along first dimension and concatenate
        one_hot = np.repeat(identity_matrix, X.shape[0], 0)
        X_concat = np.concatenate((X, one_hot), axis=3)

        return X_concat

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
