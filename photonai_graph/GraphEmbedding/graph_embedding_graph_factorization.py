from sklearn.base import BaseEstimator, TransformerMixin
from gem.embedding.gf import GraphFactorization
import numpy as np
import networkx
import os


class GraphEmbeddingGraphFactorization(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Graph Factorization (Ahmed et al., 2013).
    Graph Factorization factorizes the adjacency matrix
    with regularization. Implementation based on gem
    python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `maximum_iterations` [int, default=10000]
        the number of iterations used in sgd, when learning the embedding
    * `learning_rate` [float, default=1 * 10 ** -4]
        the learning rate of sgd
    * `regularization_coefficient` [float, default=1.0]
        the regularization coefficient for regulating the magnitude of the weights
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default=None]:
        Path to the log data



    Example
    -------
        constructor = GraphEmbeddingGraphFactorization(maximum_iterations=20000,
                                                        regularization_coefficient=0.5)
    """

    def __init__(self, embedding_dimension=1,
                 maximum_iterations=10000, learning_rate=1 * 10 ** -4,
                 regularization_coefficient=1.0,
                 construction_axis=0, adjacency_axis=0, logs=''):
        self.embedding_dimension = embedding_dimension
        self.maximum_iterations = maximum_iterations
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.construction_axis = construction_axis
        self.adjacency_axis = adjacency_axis
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
            embedding = GraphFactorization(d=self.embedding_dimension, max_iter=self.maximum_iterations,
                                           eta=self.learning_rate, regu=self.regularization_coefficient)
            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

        return embedding_list.swapaxes(0, 1)
