import unittest
import numpy as np
import dgl
from photonai_graph.GraphConversions import dgl_to_dense


class DglToDenseTest(unittest.TestCase):

    def setUp(self):
        # create dgl graph and nonsense input
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(3)
        dgl_graph.add_edges([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])
        self.graphs = [dgl_graph] * 10
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_nonsense_input(self):
        with self.assertRaises(Exception):
            dgl_to_dense(self.edge_dict)

    def test_list_type(self):
        g = dgl_to_dense(self.graphs)
        self.assertEqual(np.shape(g), (10, 3, 3))
