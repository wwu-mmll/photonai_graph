"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A collection of functions for converting graph structure formats, Visualize the different graphs,
and even create some random data to run some tests

Version
-------
Created:        15-08-2019
Last updated:   05-06-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

#TODO: make graph utility with 1. converter for GCNs, 2. converter for networkx, 3. ?
from sklearn.base import BaseEstimator, TransformerMixin
from networkx.convert_matrix import from_numpy_matrix, to_numpy_matrix
from networkx.drawing.nx_pylab import draw
from networkx.algorithms import asteroidal
import networkx as nx
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import torch
import os


def DenseToNetworkx(X, adjacency_axis = 0):

    # convert if Dense is a list of ndarrays
    if isinstance(X, list):
        graph_list = []

        for i in X:
            networkx_graph = from_numpy_matrix(A = i[:, :, adjacency_axis])
            graph_list.append(networkx_graph)

        X_converted = graph_list

    # convert if Dense is just a single ndarray
    if isinstance(X, nx.classes.graph.Graph):
        X_converted = from_numpy_matrix(A=X[:, :, adjacency_axis])

    # convert if Dense is an ndarray consisting of multiple arrays
    if isinstance(X, np.ndarray):
        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = from_numpy_matrix(A=X[i, :, :, adjacency_axis])
            graph_list.append(networkx_graph)

        X_converted = graph_list


    return X_converted


def get_random_connectivity_data(type = "dense", number_of_nodes = 114, number_of_individuals = 10, number_of_modalities = 2):
    # make random connectivity graph data
    if type == "dense":
        random_matrices = np.random.rand(number_of_individuals, number_of_nodes, number_of_nodes, number_of_modalities)

    return random_matrices


def get_random_labels(type = "classification", number_of_labels = 10):

    if type == "classification" or type == "Classification":
        y = np.random.rand(number_of_labels)
        y[y > 0.5] = 1
        y[y < 0.5] = 1

    if type == "regression" or type == "Regression":
        y = np.random.rand(number_of_labels)

    else:
        print('random labels only implemented for classification and regression. Please check your spelling')

    return y


def VisualizeNetworkx(Graphs):
    # check format in which graphs are presented or ordered
    if isinstance(Graphs, list):
        for graph in Graphs:
            draw(graph)
            plt.show()
    if isinstance(Graphs, nx.classes.graph.Graph):
        draw(Graphs)
        plt.show()
    # use networkx visualization function


def check_asteroidal(graph, return_boolean=True):

    # checks for asteroidal triples in the graph or in a list of networkx graphs
    if return_boolean:
        if isinstance(graph, list):
            graph_answer = []
            for i in graph:
                answer = asteroidal.is_at_free(i)
                graph_answer.append(answer)
        if isinstance(graph, nx.classes.graph.Graph):
            graph_answer = asteroidal.is_at_free(graph)
        else:
            print('Your input is not a networkx graph or a list of networkx graphs. Please check your inputs.')

    if not return_boolean:
        if isinstance(graph, list):
            graph_answer = []
            for i in graph:
                answer = asteroidal.find_asteroidal_triple(i)
                graph_answer.append(answer)
        if isinstance(graph, nx.classes.graph.Graph):
            graph_answer = asteroidal.find_asteroidal_triple(graph)
        else:
            print('Your input is not a networkx graph or a list of networkx graphs. Please check your inputs.')

    return graph_answer



'''
class DenseToNetworkxTransformer(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # turns a dense adjacency matrix coming from a graph constructor into a networkx graph

    def __init__(self, adjacency_axis = 0,
                 logs=''):
        self.adjacency_axis = adjacency_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X):

        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = from_numpy_matrix(A=X[i, :, :, self.adjacency_axis])
            graph_list.append(networkx_graph)

        X = graph_list

        return X


class DenseToTorchGeometricTransformer(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    # turns a dense adjacency and feature matrix coming from a graph constructor into a pytorch geometric data object

    def __init__(self, adjacency_axis = 0,
                 concatenation_axis = 3, data_as_list = 1,
                 logs=''):
        self.adjacency_axis = adjacency_axis
        self.concatenation_axis = concatenation_axis
        self.data_as_list = data_as_list
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def transform(self, X, y):

        if self.data_as_list == 1:

            # transform y to long format
            y = y.long()

            # disentangle adjacency matrix and graph
            adjacency = X[:, :, :, 1]
            feature_matrix = X[:, :, :, 0]

            # make torch tensor
            feature_matrix = torch.from_numpy(feature_matrix)
            feature_matrix = feature_matrix.float()

            # make data list for the Data_loader
            data_list = []

            # to scipy_sparse_matrix and to COO format
            for matrix in range(adjacency.shape[0]):
                # tocoo is already called in from from_scipy_sparse_matrix
                adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
                edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
                # call the right X matrix
                X_matrix = X[matrix, :, :]
                # initialize the right y value
                y_value = y[matrix]
                # build data object
                data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

            X = data_list

        return X



def GraphConverter(X, y, conversion_type = 'DenseToNetworkx', adjacency_axis = 0):

    # Convert a Dense Graph format to a Networkx format
    if conversion_type == 'DenseToNetworkx':
        #print('converting Dense to Networkx')
        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = from_numpy_matrix(A=X[i, :, :, adjacency_axis])
            graph_list.append(networkx_graph)

        X_converted = graph_list

    elif conversion_type == 'NetworkxToNumpyDense':
        #print('converting Networkx to Numpy Dense')
        graph_list = []

        for i in X:
            numpy_matrix = to_numpy_matrix(i)
            graph_list.append(numpy_matrix)

        X_converted = graph_list

    elif conversion_type == 'DenseToTorchGeometric':
        #print('converting Dense to Torch Geometric')
        if isinstance(X, list):

            # transform y to long format
            y = y.long()

            # disentangle adjacency matrix and graph
            adjacency = X[:, :, :, 1]
            feature_matrix = X[:, :, :, 0]

            # make torch tensor
            feature_matrix = torch.as_tensor(feature_matrix)
            feature_matrix = feature_matrix.float()

            # make data list for the Data_loader
            data_list = []

            # to scipy_sparse_matrix and to COO format
            for matrix in range(adjacency.shape[0]):
                # tocoo is already called in from from_scipy_sparse_matrix
                adjacency_matrix = coo_matrix(adjacency[matrix, :, :])
                edge_index, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix)
                # call the right X matrix
                X_matrix = X[matrix, :, :]
                # initialize the right y value
                y_value = y[matrix]
                # build data object
                data_list.append(Data(x=X_matrix, edge_index=edge_index, edge_attr=edge_attributes, y=y_value))

            X_converted = data_list

    elif conversion_type == 'TorchGeometricToDense':
        # Convert Dense to Torch Geometric
        print('Not implemented yet')
    elif conversion_type == 'NetworkxToTorchGeometric':
        # Convert Networkx to Torch Geometric
        print('Not implemented yet')
    elif conversion_type == 'TorchGeometricToNetworkx':
        # Convert Torch Geometric to Networkx
        print('Not implemented yet')
    elif conversion_type == 'GraphEmbeddings':
        # Convert GraphEmbeddings?
        print('Not implemented yet')
        
'''

