import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GraphConstructorPercentage(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for generating adjacency matrices 
    from connectivity matrices. Selects the top x percent
    of connections and sets all other connections to zero


    Parameters
    ----------
    * `percentage` [float]:
        value of percent of connections to discard. A value of 0.9 keeps only the top 10%
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
        constructor = GraphConstructorPercentage(percentage=0.9,
                                                 fisher_transform=1,
                                                 use_abs=1)
   """

    def __init__(self, percentage=0.8, adjacency_axis=0,
                 fisher_transform=0, use_abs=0, logs=''):
        self.percentage = percentage
        self.adjacency_axis = adjacency_axis
        self.fisher_transform = fisher_transform
        self.use_abs = use_abs
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        # generate binary matrix
        # todo: variable names should be lower case
        BinaryMatrix = np.zeros((1, X.shape[1], X.shape[2], 1))

        for i in range(X.shape[0]):
            # select top percent connections
            # calculate threshold from given percentage cutoff
            # todo: duplicated code starting here
            if np.ndim(X) == 3:
                lst = X[i, :, :].tolist()
                BinarizedMatrix = X[i, :, :].copy()
                if self.fisher_transform == 1:
                    np.arctanh(BinarizedMatrix)
                if self.use_abs == 1:
                    BinarizedMatrix = np.abs(BinarizedMatrix)
                X_transformed = X.copy()
                X_transformed = X_transformed.reshape((X_transformed.shape[0],
                                                       X_transformed.shape[1],
                                                       X_transformed.shape[2],
                                                       -1))
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
            threshold = lst[int(len(lst) * self.percentage)]

            # Threshold matrix X to create adjacency matrix
            BinarizedMatrix[BinarizedMatrix > threshold] = 1
            BinarizedMatrix[BinarizedMatrix < threshold] = 0
            BinarizedMatrix = BinarizedMatrix.reshape((-1, BinaryMatrix.shape[0], BinaryMatrix.shape[1]))
            BinarizedMatrix = BinarizedMatrix.reshape(
                (BinaryMatrix.shape[0], BinaryMatrix.shape[1], BinaryMatrix.shape[2], -1))

            # concatenate matrix back
            BinaryMatrix = np.concatenate((BinaryMatrix, BinarizedMatrix), axis=3)

        # drop first matrix as it is empty
        BinaryMatrix = np.delete(BinaryMatrix, 0, 3)
        BinaryMatrix = np.swapaxes(BinaryMatrix, 3, 0)
        X_transformed = np.concatenate((BinaryMatrix, X_transformed), axis=3)

        return X_transformed
