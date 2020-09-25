import unittest
from scipy import sparse
import numpy as np
from photonai_graph.GraphConversions import sparse_to_dense


class SparseToDenseTest(unittest.TestCase):

    def setUp(self):
        # create sparse matrices
        sparse_matrix = sparse.eye(5)
        self.single_matrix = sparse_matrix
        self.matrix_list = [sparse_matrix] * 10
        # create dense control matrix
        self.control_matrix = np.eye(5)
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_single_no_feat(self):
        g = sparse_to_dense(self.single_matrix)
        self.assertTrue(np.array_equal(g, self.control_matrix))

    def test_list_no_feat(self):
        g = sparse_to_dense(self.matrix_list)
        self.assertTrue(np.array_equal(g[0], self.control_matrix))

    def test_nonsene_no_feat(self):
        with self.assertRaises(Exception):
            sparse_to_dense(self.edge_dict)

    def test_single_feat(self):
        g = sparse_to_dense(self.single_matrix, features=self.single_matrix)
        self.assertEqual(np.shape(g), (5, 5, 2))

    def test_list_feat(self):
        g = sparse_to_dense(self.matrix_list, features=self.matrix_list)
        self.assertEqual(np.shape(g), (10, 5, 5, 2))

    def test_nonsense_feat(self):
        with self.assertRaises(Exception):
            sparse_to_dense(self.edge_dict, features=self.edge_dict)
