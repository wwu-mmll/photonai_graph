import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GraphConstructor(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Base class for all graph constructors. Implements
    methods shared by different constructors.

    Parameters
    ----------
    * `transform_style` [str, default="mean"]
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero

    """

    def __init__(self, k_distance=10,
                 transform_style="mean",
                 adjacency_axis=0, logs=''):
        self.k_distance = k_distance
        self.transform_style = transform_style
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        self.mean_matrix = None  # todo: attributes of self should be defined in the __init__

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
        # Todo: This function was marked static, but used self.
        # Todo: The indentation must be fixed (i only guessed)
        # Todo: the following two variables are used, but might be unassigned
        adjacency_matrix = None
        feature_matrix = None

        if self.transform_style == "individual":

            # ensure that the array has the "right" number of dimensions
            if np.ndim(X) == 4:
                adjacency_matrix = X[:, :, :, self.adjacency_axis].copy()
                feature_matrix = X.copy()
            # handle the case where there are 3 dimensions, meaning that there is no "adjacency axis"
            elif np.ndim(X) == 3:
                adjacency_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
                feature_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            else:
                raise Exception('input matrix needs to have 3 or 4 dimensions')

        elif self.transform_style == "mean":
            adjacency_matrix = self.mean_matrix.copy()
            if np.dim(X) == 4:
                feature_matrix = X.copy()
            elif np.dim(X) == 3:
                feature_matrix = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
        return adjacency_matrix, feature_matrix
