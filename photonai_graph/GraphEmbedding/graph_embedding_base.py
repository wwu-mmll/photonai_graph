from sklearn.base import BaseEstimator, TransformerMixin
from photonai_graph.GraphConversions import dense_to_networkx
from photonai_graph.util import assert_imported
from abc import ABC, abstractmethod
import numpy as np
import os


class GraphEmbeddingBase(BaseEstimator, TransformerMixin, ABC):
    _estimator_type = "transformer"
    """
    Base class for all embeddings. Implements helper functions
    used by other embeddings. Implementation based on gem
    python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `logs` [str, default=None]:
        Path to the log data

    """

    def __init__(self,
                 embedding_dimension: int = 1,
                 adjacency_axis: int = 0,
                 logs: str = ''):
        self.embedding_dimension = embedding_dimension
        self.adjacency_axis = adjacency_axis
        self.logs = logs
        if not self.logs:
            self.logs = os.getcwd()
        assert_imported(["gem"])

    def fit(self, X, y):
        return self

    @staticmethod
    def _calculate_embedding(embedding, matrices):
        """Returns embedding of graphs"""
        graphs = dense_to_networkx(matrices)  # convert matrices
        embedding_list = []

        for graph in graphs:
            embedding.learn_embedding(graph=graph, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            embedding_list.append(embedding_representation)

        embedding_list = np.squeeze(np.asarray(embedding_list))

        return embedding_list

    def transform(self, X):
        """Transforms graph using Laplacian Eigenmaps Embedding"""
        embedding = self._init_embedding()
        X_transformed = self._calculate_embedding(embedding, X)
        X_transformed = np.real(X_transformed)
        return X_transformed

    @abstractmethod
    def _init_embedding(self):
        """Initilaize the current embedding"""
