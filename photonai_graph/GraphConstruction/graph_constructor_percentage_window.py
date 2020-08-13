import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from photonai_graph.GraphUtilities import individual_ztransform, individual_fishertransform


class GraphConstructorPercentageWindow(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the top x percent
    of connections and sets all other connections to zero


    Parameters
    ----------
    * `percentage_upper` [float]:
        upper limit of the percentage window
    * `percentage_lower` [float]:
        lower limit of the percentage window
    * `adjacency_axis` [int]:
        position of the adjacency matrix, default being zero
    * `one_hot_nodes` [int]:
        Whether to generate a one hot encoding of the nodes in the matrix.
    * `return_adjacency_only` [int]:
        whether to return the adjacency matrix only (1) or also a feature matrix (0)
    * `fisher_transform` [int]:
        Perform a fisher transform of each matrix. No (0) or Yes (1)
    * `use_abs` [int]:
        Changes the values to absolute values. Is applied after fisher transform and before
        z-score transformation
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default='']:
        Path to the log data

    Example
    -------
        constructor = GraphConstructorPercentageWindow(percentage=0.9,
                                                       fisher_transform=1,
                                                       use_abs=1)
   """

    def __init__(self, transform_style: str = "individual",
                 percentage_upper=0.5,
                 percentage_lower=0.1,
                 adjacency_axis=0,
                 concatenation_axis=3,
                 one_hot_nodes=0,
                 return_adjacency_only=0,
                 fisher_transform=0,
                 use_abs=0,
                 zscore=1,
                 use_abs_zscore=0,
                 logs=''):
        self.transform_style = transform_style
        self.percentage_upper = percentage_upper
        self.percentage_lower = percentage_lower
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
        # todo: ...
        pass

    def transform(self, X):
        # todo: duplicated code
        # ensure that the array has the "right" number of dimensions
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
        if self.transform_style == "individual":
            for i in range(X.shape[0]):
                # select top percent connections
                # calculate threshold from given percentage cutoff
                if np.ndim(X) == 3:
                    lst = X[i, :, :].tolist()
                    BinarizedMatrix = X[i, :, :].copy()
                    if self.fisher_transform == 1:
                        np.arctanh(BinarizedMatrix)
                    if self.use_abs == 1:
                        BinarizedMatrix = np.abs(BinarizedMatrix)
                    X_transformed = X.copy()
                    X_transformed = X_transformed.reshape(
                        (X_transformed.shape[0], X_transformed.shape[1], X_transformed.shape[2], -1))
                elif np.ndim(X) == 4:
                    lst = X[i, :, :, self.adjacency_axis].tolist()
                    BinarizedMatrix = X[i, :, :, self.adjacency_axis].copy()
                    if self.fisher_transform == 0:
                        np.arctanh(BinarizedMatrix)
                    if self.use_abs == 1:
                        BinarizedMatrix = np.abs(BinarizedMatrix)
                    X_transformed = X.copy()
                else:
                    raise ValueError('Input matrix needs to have either 3 or 4 dimensions not more or less.')
                lst = [item for sublist in lst for item in sublist]
                lst.sort()
                # new_lst = lst[int(len(lst) * self.percentage): int(len(lst) * 1)]
                # threshold = new_lst[0]
                threshold_upper = lst[int(len(lst) * self.percentage_upper)]
                threshold_lower = lst[int(len(lst) * self.percentage_lower)]

                # Threshold matrix X to create adjacency matrix
                BinarizedMatrix[BinarizedMatrix > threshold_upper] = 0
                BinarizedMatrix[BinarizedMatrix < threshold_upper] = 1
                BinarizedMatrix[BinarizedMatrix < threshold_lower] = 0
                BinarizedMatrix = BinarizedMatrix.reshape((-1, BinaryMatrix.shape[0], BinaryMatrix.shape[1]))
                BinarizedMatrix = BinarizedMatrix.reshape(
                    (BinaryMatrix.shape[0], BinaryMatrix.shape[1], BinaryMatrix.shape[2], -1))

                # concatenate matrix back
                BinaryMatrix = np.concatenate((BinaryMatrix, BinarizedMatrix), axis=3)

                # drop first matrix as it is empty
            BinaryMatrix = np.delete(BinaryMatrix, 0, 3)
            BinaryMatrix = np.swapaxes(BinaryMatrix, 3, 0)
            X_transformed = np.concatenate((BinaryMatrix, X_transformed), axis=3)
        else:
            raise Exception('Transformer only implemented for individual transform')

        return X_transformed
