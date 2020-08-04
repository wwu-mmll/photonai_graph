"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper handling the different embedding strategies to perform photonai_graph embedding

Version
-------
Created:        28-08-2019
Last updated:   04-06-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

import os
import networkx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.sdne import SDNE

# TODO: add documentation for every method

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


    def __init__(self, embedding_dimension = 1,
                 maximum_iterations = 10000, learning_rate = 1 * 10 ** -4,
                 regularization_coefficient = 1.0,
                 construction_axis = 0, adjacency_axis = 0, logs=''):
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
            #take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            #transform this photonai_graph with a photonai_graph embedding
            embedding = GraphFactorization(d=self.embedding_dimension, max_iter=self.maximum_iterations,
                                       eta=self.learning_rate, regu=self.regularization_coefficient)
            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

        return embedding_list.swapaxes(0, 1)



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


    def __init__(self, embedding_dimension = 1,
                 decay_factor = 0.01,
                 construction_axis = 0, adjacency_axis = 0, logs=''):
        self.embedding_dimension = embedding_dimension
        self.decay_factor = decay_factor
        self.construction_axis = construction_axis
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        for i in range(X.shape[0]):
            #take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            #transform this photonai_graph with a photonai_graph embedding
            if self.embedding_dimension == 1:
                embedding = HOPE(d=2, beta=self.decay_factor)
                embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
                embedding_representation = embedding.get_embedding()
                embedding_representation = np.reshape(embedding_representation, (-1, 1))

            else:
                embedding = HOPE(d=self.embedding_dimension, beta=self.decay_factor)
                embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
                embedding_representation = embedding.get_embedding()
                embedding_representation = np.reshape(embedding_representation, (embedding_representation.shape[0], embedding_representation.shape[1], -1))

            if 'embedding_list' not in locals():
                embedding_list = embedding_representation
            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)


        return embedding_list.swapaxes(0, 1)



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


    def __init__(self, embedding_dimension = 1,
                 decay_factor = 0.01,
                 construction_axis = 0, adjacency_axis = 0, logs=''):
        self.embedding_dimension = embedding_dimension
        self.adjacency_axis = adjacency_axis
        self.construction_axis = construction_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        for i in range(X.shape[0]):
            #take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            #transform this photonai_graph with a photonai_graph embedding
            embedding = LaplacianEigenmaps(d=self.embedding_dimension)
            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

            print(embedding_list.shape)

        return embedding_list.swapaxes(0, 1)




class GraphEmbeddingLocallyLinearEmbedding(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Locally Linear Embedding (Roweis & Saul, 
    2000).
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

    def __init__(self, embedding_dimension = 1,
                 decay_factor = 0.01,
                 construction_axis = 0, adjacency_axis = 0, logs=''):
        self.embedding_dimension = embedding_dimension
        self.adjacency_axis = adjacency_axis
        self.construction_axis = construction_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        for i in range(X.shape[0]):
            #take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            #transform this photonai_graph with a photonai_graph embedding
            embedding = LocallyLinearEmbedding(d=self.embedding_dimension)
            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

        return embedding_list.swapaxes(0, 1)



class GraphEmbeddingSDNE(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Structural Deep Network Embedding (Wang, Cui
    & Zhu, 2016).
    Implementation based on gem python package.
    

    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `seen_edge_reconstruction_weight` [int, default=5]
	the penalty parameter beta in matrix B of the 2nd order objective
    * `first_order_proximity_weight` [float, default=1e-5]
	the weighing hyperparameter alpha for the 1st order objective
    * `lasso_regularization_coefficient` [float, default=1e-6]
	the L1 regularization coefficient
    * `ridge_regression_coefficient` [float, default=1e-6]
	the L2 regularization coefficient
    * `number_of_hidden_layers` [int, default=3]
	the number of hidden layers in the encoder/decoder
    * `layer_sizes` [int, default=[50, 15,]]
	the number of units per layer in the hidden layers of the encoder/decoder. Vector of length number_of_hidden_layers
	-1
    * `num_iterations` [int, default=50]
	the number of iterations with which to train the network
    * `learning_rate` [float, default=0.01]
	the learning rate with which the network is trained
    * `batch_size` [int, default=500]
	the batch size when training the algorithm
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `verbosity` [int, default=0]:
        The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug
    * `logs` [str, default=None]:
        Path to the log data
    
    

    Example
    -------
        constructor = GraphEmbeddingSDNE(embedding_dimension=1,
                            		 seen_edge_reconstruction_weight=10,
					 first_order_proximity_weight=1e-4
					 num_hidden_layers=5,
					 layer_sizes=[50, 25, 20, 15,])
    """
    def __init__(self, embedding_dimension = 1,
                 seen_edge_reconstruction_weight = 5,
                 first_order_proximity_weight = 1e-5,
                 lasso_regularization_coefficient = 1e-6,
                 ridge_regression_coefficient = 1e-6,
                 construction_axis = 0,
                 number_of_hidden_layers = 3,
                 layer_sizes = [50, 15,],
                 num_iterations = 50,
                 learning_rate  = 0.01,
                 batch_size = 500,
                 adjacency_axis = 0,
                 logs=''):
        self.embedding_dimension = embedding_dimension
        self.seen_edge_reconstruction_weight = seen_edge_reconstruction_weight
        self.first_order_proximity_weight = first_order_proximity_weight
        self.lasso_regularization_coefficient = lasso_regularization_coefficient
        self.ridge_regression_coefficient = ridge_regression_coefficient
        self.number_of_hidden_layers = number_of_hidden_layers
        self.layer_sizes = layer_sizes
        self.num_iterations = num_iterations
        self.construction_axis = construction_axis
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        for i in range(X.shape[0]):
            #take matrix and transform it into photonai_graph
            G = networkx.convert_matrix.from_numpy_matrix(X[i, :, :, self.adjacency_axis])

            embedding = SDNE(d= self.embedding_dimension, beta=self.seen_edge_reconstruction_weight,
                             alpha=self.first_order_proximity_weight, nu1=self.lasso_regularization_coefficient,
                             nu2=self.ridge_regression_coefficient, K=self.number_of_hidden_layers,
                             n_units=self.layer_sizes, n_iter=self.num_iterations,
                             xeta=self.learning_rate, n_batch=self.batch_size,
                modelfile=['enc_model.json', 'dec_model.json'],
                weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])

            embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            embedding_representation = embedding.get_embedding()
            if 'embedding_list' not in locals():
                embedding_list = embedding_representation

            else:
                embedding_list = np.concatenate((embedding_list, embedding_representation), axis=-1)

            print(embedding_list.shape)

        return embedding_list.swapaxes(0, 1)
