from sklearn.base import BaseEstimator, TransformerMixin
from gem.embedding.hope import HOPE
import numpy as np
import networkx
import os


# todo: why does this class not inherit from GraphEmbeddingBase?
class GraphEmbeddingHOPE(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Higher-order proximity preserved embedding
    (Mingdong et al., 2016).
    Implementation based on gem python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `decay_factor` [ifloat, default=0.01]
        the higher order coefficient beta
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default=None]:
        Path to the log data



    Example
    -------
        constructor = GraphEmbeddingHOPE(embedding_dimension=1,
                                          decay_factor=0.1)
    """

    def __init__(self,
                 embedding_dimension=1,
                 decay_factor=0.01,
                 construction_axis=0,
                 adjacency_axis=0,
                 logs=''):
        self.embedding_dimension = embedding_dimension
        self.decay_factor = decay_factor
        self.construction_axis = construction_axis
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X):

        embedding_list = []

        if X.shape[0] == 0:
            raise ValueError("Check the shape of your input.\nReceived X.shape[0]==0..?")

        for i in range(X.shape[0]):
            # take matrix and transform it into photonai_graph
            g = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            # transform this photonai_graph with a photonai_graph embedding
            if self.embedding_dimension == 1:
                embedding = HOPE(d=2, beta=self.decay_factor)
                embedding.learn_embedding(graph=g, is_weighted=True, no_python=True)
                embedding_representation = embedding.get_embedding()
                embedding_representation = np.reshape(embedding_representation, (-1, 1))

            else:
                embedding = HOPE(d=self.embedding_dimension, beta=self.decay_factor)
                embedding.learn_embedding(graph=g, is_weighted=True, no_python=True)
                embedding_representation = embedding.get_embedding()
                embedding_representation = np.reshape(embedding_representation,
                                                      (embedding_representation.shape[0],
                                                       embedding_representation.shape[1],
                                                       -1))

            embedding_list.append(embedding_representation)

        return np.asarray(embedding_list)
