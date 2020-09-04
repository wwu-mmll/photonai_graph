from sklearn.base import BaseEstimator, TransformerMixin
from photonai_graph.GraphConversions import dense_to_networkx
from abc import ABC
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
                 embedding_dimension=1,
                 adjacency_axis: int = 0,
                 logs: str = ''):
        self.embedding_dimension = embedding_dimension
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    @staticmethod
    def calculate_embedding(embedding, matrices):
        """Returns embedding of graphs"""
        graphs = dense_to_networkx(matrices)  # convert matrices
        embedding_list = []

        for graph in graphs:
            embedding.learn_embedding(graph=graph, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

        return embedding_list.swapaxes(0, 1)

