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

    def test_disconnected_graph(self):
        in_graph = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        in_feats = np.array([[.9, .3, .2], [.8, .2, .1], [.7, .1, .0]])
        in_graph = np.expand_dims(in_graph, axis=(0, -1))
        in_feats = np.expand_dims(in_feats, axis=(0, -1))
        in_graph = np.concatenate((in_graph, in_feats), axis=-1)
        dgl_graph = dense_to_dgl(in_graph, adjacency_axis=0, feature_axis=1)[0]
        self.assertEqual(dgl_graph.num_nodes(), 3)
        self.assertEqual(dgl_graph.num_edges(), 2)
        self.assertTrue(np.array_equal(dgl_graph.ndata['feat'].numpy(), in_feats[0, ..., 0]))
