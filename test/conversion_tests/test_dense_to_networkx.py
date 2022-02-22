import unittest
import networkx as nx
import numpy as np
from photonai_graph.GraphConversions import dense_to_networkx


class DenseToNetworkxTest(unittest.TestCase):

    def setUp(self):
        self.random4d = np.random.rand(20, 20, 20, 2)
        self.single4d = np.random.rand(1, 20, 20, 2)
        self.array_list = [np.random.rand(20, 20, 2)] * 10
        self.edge_dict = self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_output_list(self):
        mtrx = dense_to_networkx(self.random4d)
        self.assertEqual(type(mtrx), list)

    def test_output_list_element(self):
        mtrx = dense_to_networkx(self.random4d)
        self.assertEqual(type(mtrx[0]), nx.Graph)

    def test_input_single_dense(self):
        mtrx = dense_to_networkx(self.single4d)
        self.assertEqual(type(mtrx), nx.Graph)

    def test_input_list_dense(self):
        mtrx = dense_to_networkx(self.array_list)
        self.assertEqual(type(mtrx[0]), nx.Graph)

    def test_input_nonsense(self):
        with self.assertRaises(ValueError):
            dense_to_networkx(self.edge_dict)
