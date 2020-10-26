import unittest
import numpy as np
import scipy.sparse
import networkx as nx
from photonai_graph.GraphUtilities import draw_connectivity_matrix


class DrawConnectivityMatrixTest(unittest.TestCase):

    def setUp(self):
        # numpy matrices
        self.m2d_dense = np.random.rand(20, 20)
        self.m3d_dense = np.random.rand(10, 20, 20)
        self.m3d_dense_adjacency = np.random.rand(20, 20, 2)
        self.m3d_dense_adjacency_list = [np.random.rand(20, 20, 2)] * 10
        self.m4d_dense = np.random.rand(10, 20, 20, 2)
        self.m5d_dense = np.random.rand(10, 20, 20, 2, 2)
        # sparse matrices
        self.m2d_sparse = [scipy.sparse.rand(20, 20)] * 5
        # networkx graphs
        self.nx_graph_single = nx.erdos_renyi_graph(20, p=0.3)
        self.nx_graphs = [nx.erdos_renyi_graph(20, p=0.3)] * 10

    def test_2d_dense(self):
        draw_connectivity_matrix(self.m2d_dense, show=False)

    def test_2d_colorbar(self):
        draw_connectivity_matrix(self.m2d_dense, colorbar=True, show=False)

    def test_2d_adjacency_axis(self):
        with self.assertRaises(Exception):
            draw_connectivity_matrix(self.m2d_dense, adjacency_axis=0, show=False)

    def test_3d_dense(self):
        draw_connectivity_matrix(self.m3d_dense, show=False)

    def test_3d_dense_colorbar(self):
        draw_connectivity_matrix(self.m3d_dense, colorbar=True, show=False)

    def test_3d_dense_adjacency(self):
        draw_connectivity_matrix(self.m3d_dense_adjacency, colorbar=True, adjacency_axis=0, show=False)

    def test_3d_dense_adjacency_list(self):
        draw_connectivity_matrix(self.m3d_dense_adjacency_list, show=False)

    def test_3d_dense_adjacency_list_specified(self):
        draw_connectivity_matrix(self.m3d_dense_adjacency_list, colorbar=True,
                                 adjacency_axis=0, show=False)

    def test_4d_dense(self):
        with self.assertRaises(Exception):
            draw_connectivity_matrix(self.m4d_dense, show=False)

    def test_4d_dense_adjacency(self):
        draw_connectivity_matrix(self.m4d_dense, adjacency_axis=0, show=False)

    def test_4d_dense_colorbar(self):
        draw_connectivity_matrix(self.m4d_dense, colorbar=True, adjacency_axis=0, show=False)

    def test_5d_dense_no_adjacency(self):
        with self.assertRaises(Exception):
            draw_connectivity_matrix(self.m5d_dense)

    def test_2d_sparse(self):
        draw_connectivity_matrix(self.m2d_sparse, show=False)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            draw_connectivity_matrix(self.nx_graphs, show=False)

    def test_wrong_type_single(self):
        with self.assertRaises(TypeError):
            draw_connectivity_matrix(self.nx_graph_single, show=False)
