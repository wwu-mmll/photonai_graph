import unittest
import networkx as nx
import numpy as np
from photonai_graph.GraphConversions import networkx_to_dense


class NetworkxToDenseTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.cycle_graph(20)] * 10
        self.check_graph = nx.cycle_graph(5)
        self.check_matrix = np.array([[0, 1, 0, 0, 1],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 0, 1, 0, 1],
                                      [1, 0, 0, 1, 0]])

    def test_list(self):
        mtrx = networkx_to_dense(self.graphs)
        self.assertEqual(type(mtrx), list)

    def test_np_type(self):
        mtrx = networkx_to_dense(self.graphs)
        self.assertEqual(type(mtrx[0]), np.ndarray)

    def test_single_graph(self):
        mtrx = networkx_to_dense(self.graph)
        self.assertEqual(type(mtrx), np.ndarray)

    def test_matrix_shape(self):
        mtrx = networkx_to_dense(self.graph)
        self.assertEqual(np.shape(mtrx), (20, 20))

    def test_check_graph(self):
        mtrx = networkx_to_dense(self.check_graph)
        self.assertTrue(np.array_equal(mtrx, self.check_matrix))

    def test_check_nonsense_input(self):
        with self.assertRaises(ValueError):
            mtrx = networkx_to_dense(self.check_matrix)
