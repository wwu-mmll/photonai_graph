import unittest
import numpy as np
import igraph
from photonai_graph.GraphConversions import dense_to_igraph


class DenseToIgraphTest(unittest.TestCase):

    def setUp(self):
        self.random4d = np.random.rand(20, 20, 20, 2)

    def test_axis_argument(self):
        with self.assertRaises(NotImplementedError):
            dense_to_igraph(self.random4d, feature_axis=1)

    def test_matrix(self):
        g = dense_to_igraph(self.random4d, adjacency_axis=0)
        self.assertEqual(type(g[0]), igraph.Graph)
        self.assertEqual(type(g), list)

    def test_node_count(self):
        g = dense_to_igraph(self.random4d, adjacency_axis=0)
        self.assertEqual(g[0].vcount(), 20)
