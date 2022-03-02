import unittest
import networkx as nx
import numpy as np
from photonai_graph.GraphConversions import dense_to_networkx


class DenseToNetworkxTest(unittest.TestCase):

    def setUp(self):
        self.random4d = np.random.rand(20, 20, 20, 2)
        self.array_list = [np.random.rand(20, 20, 2)] * 10
        self.edge_dict = self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_output_list(self):
        mtrx = dense_to_networkx(self.random4d)
        self.assertEqual(type(mtrx), list)
        for itm in mtrx:
            self.assertEqual(type(itm), nx.Graph)

    def test_input_list_dense(self):
        mtrx = dense_to_networkx(self.array_list)
        self.assertEqual(type(mtrx[0]), nx.Graph)

    def test_input_nonsense_list(self):
        with self.assertRaises(ValueError):
            dense_to_networkx([self.edge_dict] * 10)

    def test_input_nonsense_array(self):
        with self.assertRaises(ValueError):
            dense_to_networkx(np.array([self.edge_dict] * 10))

    def test_input_nonsense(self):
        with self.assertRaises(ValueError):
            dense_to_networkx(self.edge_dict)
