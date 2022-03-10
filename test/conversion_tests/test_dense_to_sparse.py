import unittest
import numpy as np
from scipy import sparse
from photonai_graph.GraphConversions import dense_to_sparse


class DenseToSparseTest(unittest.TestCase):

    def setUp(self):
        # create dense matrices
        id_matrix = np.eye(3)
        id_array = id_matrix[:, :, np.newaxis]
        id_3d_no_adj = id_matrix[np.newaxis, :, :]
        id_3d = np.repeat(id_array, 2, axis=2)
        id_3d_no_adj = np.repeat(id_3d_no_adj, 5, axis=0)
        id_4d = np.repeat(id_3d[np.newaxis, :, :, :], 5, axis=0)
        self.matrix_2d = id_matrix
        self.matrix_2d_list = [id_matrix] * 10
        self.matrix_3d = id_3d
        self.matrix_3d_list = [id_3d] * 10
        self.matrix_3d_no_adj = id_3d_no_adj
        self.matrix_4d = id_4d
        self.matrix_4d_list = [id_4d] * 10
        # create sparse control matrices
        sparse_matrix = sparse.eye(3)
        self.sparse_2d = sparse_matrix
        # create nonsense input
        self.edge_dict = self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_wrong_type(self):
        with self.assertRaises(KeyError):
            dense_to_sparse(self.matrix_3d, m_type="xxl_matrix")

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            dense_to_sparse(self.edge_dict)

    def test_2d_matrix(self):
        mtrx = dense_to_sparse([self.matrix_2d])
        self.assertTrue((mtrx[0][0] != self.sparse_2d).nnz == 0)

    def test_2d_list(self):
        mtrx = dense_to_sparse(self.matrix_2d_list)
        adj_list = mtrx[0]
        self.assertTrue((adj_list[0] != self.sparse_2d).nnz == 0)

    def test_3d_matrix_no_adj(self):
        mtrx = dense_to_sparse(self.matrix_3d_no_adj)
        adj_list = mtrx[0]
        self.assertTrue((adj_list[0] != self.sparse_2d).nnz == 0)

    def test_3d_adj(self):
        mtrx = dense_to_sparse([self.matrix_3d], adjacency_axis=0, feature_axis=1)
        self.assertTrue((mtrx[0][0] != self.sparse_2d).nnz == 0)

    def test_3d_matrix_list(self):
        mtrx = dense_to_sparse(self.matrix_3d_list, adjacency_axis=0, feature_axis=1)
        adj_list = mtrx[0]
        self.assertTrue((adj_list[0] != self.sparse_2d).nnz == 0)

    def test_4d_no_adj(self):
        with self.assertRaises(ValueError):
            dense_to_sparse([self.matrix_4d])

    def test_4d_adj(self):
        mtrx = dense_to_sparse(self.matrix_4d, adjacency_axis=0, feature_axis=0)
        adj_list = mtrx[0]
        self.assertTrue((adj_list[0] != self.sparse_2d).nnz == 0)

    def test_4d_list(self):
        with self.assertRaises(ValueError):
            dense_to_sparse(self.matrix_4d_list, adjacency_axis=0, feature_axis=1)

    def test_nonsense_input(self):
        with self.assertRaises(ValueError):
            dense_to_sparse(self.edge_dict)
