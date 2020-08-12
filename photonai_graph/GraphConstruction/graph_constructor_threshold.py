import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform


class GraphConstructorThreshold(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Thresholds matrix based
    on a chosen threshold value.


    Parameters
    ----------
    * `threshold` [float]:
        threshold value below which to set matrix entries to zero
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
        constructor = GraphConstructorThreshold(threshold=0.5,
                                                fisher_transform=1,
                                                use_abs=1)
   """

    def __init__(self, threshold=0.1, adjacency_axis=0,
                 concatenation_axis=3,
                 one_hot_nodes=0,
                 return_adjacency_only=0,
                 fisher_transform=0,
                 use_abs=0,
                 zscore=1,
                 use_abs_zscore=0,
                 logs=''):
        self.threshold = threshold
        self.adjacency_axis = adjacency_axis
        self.concatenation_axis = concatenation_axis
        self.one_hot_nodes = one_hot_nodes
        self.return_adjacency_only = return_adjacency_only
        self.fisher_transform = fisher_transform
        self.use_abs = use_abs
        self.zscore = zscore
        self.use_abs_zscore = use_abs_zscore
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        # ensure that the array has the "right" number of dimensions
        # todo: duplicated code
        # todo: all variable names should be lowercase (with an exeption for X)
        if np.ndim(X) == 4:
            Threshold_matrix = X[:, :, :, self.adjacency_axis].copy()
            X_transformed = X.copy()
            if self.fisher_transform == 1:
                Threshold_matrix = individual_fishertransform(Threshold_matrix)
            if self.use_abs == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
            if self.zscore == 1:
                Threshold_matrix = individual_ztransform(Threshold_matrix)
            if self.use_abs_zscore == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
        # handle the case where there are 3 dimensions, meaning that there is no "adjacency axis"
        elif np.ndim(X) == 3:
            Threshold_matrix = X.copy()
            X_transformed = X.copy().reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            if self.fisher_transform == 1:
                Threshold_matrix = individual_fishertransform(Threshold_matrix)
            if self.use_abs == 1:
                Threshold_matrix = np.abs(Threshold_matrix)
            if self.zscore == 1:
                Threshold_matrix = individual_ztransform(Threshold_matrix)
            if self.use_abs_zscore == 1:
                Threshold_matrix = np.abs(Threshold_matrix)

        else:
            raise Exception('encountered unusual dimensions, please check your dimensions')
        # This creates and indvidual adjacency matrix for each person
        # todo: threshold missing equal values (<= or >=)
        Threshold_matrix[Threshold_matrix > self.threshold] = 1
        # todo: duplicated code starting here again
        Threshold_matrix[Threshold_matrix < self.threshold] = 0
        # add extra dimension to make sure that concatenation works later on
        Threshold_matrix = Threshold_matrix.reshape(Threshold_matrix.shape[0], Threshold_matrix.shape[1],
                                                    Threshold_matrix.shape[2], -1)

        # Add the matrix back again
        if self.one_hot_nodes == 1:
            # construct an identity matrix
            identity_matrix = np.identity((X.shape[1]))
            # expand its dimension for later re-addition
            identity_matrix = np.reshape(identity_matrix, (-1, identity_matrix.shape[0], identity_matrix.shape[1]))
            identity_matrix = np.reshape(identity_matrix, (
            identity_matrix.shape[0], identity_matrix.shape[1], identity_matrix.shape[2], -1))
            one_hot_node_features = np.repeat(identity_matrix, X.shape[0], 0)
            # concatenate matrices
            X_transformed = np.concatenate((Threshold_matrix, one_hot_node_features), axis=self.concatenation_axis)
        else:
            if self.return_adjacency_only == 0:
                X_transformed = np.concatenate((Threshold_matrix, X_transformed), axis=self.concatenation_axis)
            elif self.return_adjacency_only == 1:
                X_transformed = Threshold_matrix.copy()
            else:
                return ValueError(
                    "The argument return_adjacency_only takes only values 0 or 1 no other values. Please check your input values")
            # X_transformed = np.delete(X_transformed, self.adjacency_axis, self.concatenation_axis)

        return X_transformed
