from photonai_graph.DynamicUtils.cofluct_functions import cofluct
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os


class CofluctTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 quantiles: tuple = (0, 1),
                 return_mat: bool = True,
                 adjacency_axis: int = 0,
                 logs=None):
        """
        Class for calculating time series co-activation. Based on
        Esfahlani et al, 2020.

        Parameters
        ----------
        quantiles: tuple,default=(0, 1)
            lower and upper bound of connection strength quantile to look at.
        return_mat: bool,default=True
            Whether to return matrix (True) or vector (False).
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero

        Example
        -------
        ```python
            transformer = CofluctTransform(quantiles=(0.95, 1), return_mat=True)
        ```
        """
        self.quantiles = quantiles
        self.return_mat = return_mat
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        # check if matrix is 3d or 4d
        if np.ndim(X) == 3:
            mtrx = X.copy()
        elif np.ndim(X) == 4:
            mtrx = X[:, :, :, self.adjacency_axis]
        else:
            raise TypeError("Matrix needs to 3d or 4d."
                            "Individuals x Nodes x Timepoints (x Modalities).")

        # calculate cofluct for each matrix
        cfls = []

        for m in range(mtrx.shape[0]):
            ser = cofluct(mtrx[m, :, :], self.quantiles, self.return_mat)
            cfls.append(ser)

        np.asarray(cfls)

        return cfls
