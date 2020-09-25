import unittest
from scipy import sparse
import numpy as np
import dgl
from photonai_graph.GraphConversions import sparse_to_dgl


class SparseToDglTest(unittest.TestCase):

    def setUp(self):
        # create sparse matrices
        sparse_matrix = sparse.eye(5)
        sparse_list = [sparse_matrix] * 10
        self.single_matrix = sparse_matrix
        self.matrix_list = sparse_list
        self.sparse_tuple = (sparse_list, sparse_list)
        # create dense control matrix
        self.control_matrix = np.eye(5)
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_tuple(self):
        g = sparse_to_dgl(self.sparse_tuple, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(g), list)

    def test_tuple_graph(self):
        g = sparse_to_dgl(self.sparse_tuple, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(g[0]), dgl.DGLGraph)

    def test_list(self):
        g = sparse_to_dgl(self.matrix_list)
        self.assertEqual(type(g), list)

    def test_list_graph(self):
        g = sparse_to_dgl(self.matrix_list)
        self.assertEqual(type(g[0]), dgl.DGLGraph)

    def test_nonsense(self):
        with self.assertRaises(Exception):
            sparse_to_dgl(self.edge_dict)
