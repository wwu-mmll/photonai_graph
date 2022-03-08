from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Union
import numpy as np


class PopulationAveragingTransform(BaseEstimator, TransformerMixin):
    def __init__(self,
                 adjacency_axis: int = 0,
                 connectivity_axis: int = 1,
                 feature_axis: Union[int, Tuple[int, ...]] = 2):
        """Transformation for averaging over the population
        The connectivity of an individual is discarded by this transformation.
        Instead, the precalculated mean is used and the original connectivity will be passed as features.

        !!! warning
            The original adjacency matrix of unseen data is discarded by application of this transformer!

        Parameters
        ----------
        adjacency_axis: int,default=0
            Index of the adjacency part of the input arrays.
            If a GraphConstructor is used with default parameters the adjacency is stored at the first dimension.
        connectivity_axis: int,default=1
            Index of the connectivity part of the input arrays for unseen data.
            If a GraphConstructor is used with default parameters the connectivity is at the second dimension.
        feature_axis: Union[int, List[int, ...]],default=2
            Index (or indices) of the feature part of the unseen data.

        Example
        -------
        ```python
        my_pipe.add(PipelineElement('PopulationAveragingTransform'))
        ```

        Notes
        -----
        Using the feature axis as additional feature is not implemented yet.
        """
        self.connectivtiy_axis = connectivity_axis
        self.adjacency_axis = adjacency_axis
        self.feature_axis = feature_axis
        self.learned_mean = None
        if self.feature_axis != 2:
            raise NotImplementedError("Feature passing is not implemented yet.")

    def fit(self, X, y):
        if isinstance(X, list):
            x_int = np.asarray(X)
        else:
            x_int = X
        mean = np.mean(x_int, axis=0)
        if np.ndim(x_int) == 4:
            self.learned_mean = mean[..., self.adjacency_axis]
        elif np.ndim(x_int) == 3:
            self.learned_mean = mean
        else:
            raise ValueError("The input matrices need to have 3 or 4 dimensions. Please check your input matrix.")
        return self

    def transform(self, X):
        if self.learned_mean is None:
            raise ValueError("This transformer has not been fitted yet.")
        if np.ndim(X) < 3:
            raise ValueError("Usage of population averaging without features or original connectivity is discouraged.\n"
                             "Please read the documentation of this transformer")
        if isinstance(X, list):
            x_int = np.array(X)
        else:
            x_int = X
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Expected list or np.ndarray as input type, got {type(X)}")
        new_adjacency = self.learned_mean.copy()
        new_adjacency = new_adjacency[np.newaxis, ..., np.newaxis]
        new_adjacency = np.repeat(new_adjacency, x_int.shape[0], axis=0)
        return np.concatenate((new_adjacency, x_int[..., self.feature_axis, np.newaxis]), axis=3)
