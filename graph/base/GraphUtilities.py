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
import torch
from networkx.convert_matrix import from_numpy_matrix, to_numpy_matrix
from networkx.drawing.nx_pylab import draw
import networkx.drawing as drawx
from networkx.algorithms import asteroidal
import networkx as nx
import pygraphviz
import numpy as np
import pydot
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from photonai.base import PhotonRegistry
import os
import json


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



def pygraphviz_to_nx(Graphs):
    if isinstance(Graphs, list):
        A_Graphs = []
        for graph in Graphs:
            a_graph = drawx.nx_agraph.from_agraph(graph)
            A_Graphs.append(a_graph)

    elif isinstance(Graphs, pygraphviz.agraph.AGraph):
        A_Graphs = drawx.nx_agraph.from_agraph(Graphs)

    else:
        raise TypeError('The input needs to be list of pygraphviz files or a single pygraphviz file. Please check your inputs.')

    return A_Graphs

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


def RegisterGraph_force():

    registry = PhotonRegistry()

    base_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    BaseJSON = os.path.join(base_folder, 'photonai/base/registry/PhotonCore.json')
    GraphJSON = os.path.join(base_folder, 'photonai/graph/base/PhotonGraph.json')

    # if a graph element is not registered
    if not registry.check_availability("GraphConstructorPercentage"):
        print('Graph available in a sec')
        with open(BaseJSON, 'r') as base_json_file, open(GraphJSON, 'r') as graph_json_file:
            base_j = json.load(base_json_file)
            graph_j = json.load(graph_json_file)
        base_j.update(graph_j)

        with open(BaseJSON, 'w') as tf:
            json.dump(base_j, tf)

    # if a graph element is already registered
    else:
        print('Graph already available')

    return print('done')

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

