import unittest
import numpy as np
import dgl
from photonai_graph.GraphConversions import dense_to_dgl


class DenseToDglTest(unittest.TestCase):

    def setUp(self):
        self.random4d = np.random.rand(20, 20, 20, 2)
        self.edge_dict = self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_axis_argument(self):
        with self.assertRaises(NotImplementedError):
            dense_to_dgl(self.random4d)

    def test_matrix(self):
        g = dense_to_dgl(self.random4d, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(g[0]), dgl.DGLGraph)

    def test_nonsense_input(self):
        with self.assertRaises(ValueError):
            dense_to_dgl(self.edge_dict, adjacency_axis=0, feature_axis=1)
