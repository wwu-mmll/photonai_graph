from sklearn.base import BaseEstimator, TransformerMixin
from gem.embedding.lap import LaplacianEigenmaps
import numpy as np
import networkx
import os


class GraphEmbeddingLaplacianEigenmaps(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Laplacian Eigenmaps (Belkin & Niyogi, 2013).
    Implementation based on gem python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default=None]:
        Path to the log data



    Example
    -------
        constructor = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1,
                                                       decay_factor=0.1)
    """

    def __init__(self, embedding_dimension=1,
                 decay_factor=0.01,
                 construction_axis=0, adjacency_axis=0, logs=''):
        self.embedding_dimension = embedding_dimension
        self.adjacency_axis = adjacency_axis
        self.construction_axis = construction_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X):

        for i in range(X.shape[0]):
            # take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            # transform this photonai_graph with a photonai_graph embedding
            embedding = LaplacianEigenmaps(d=self.embedding_dimension)
            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

            print(embedding_list.shape)

        return embedding_list.swapaxes(0, 1)
