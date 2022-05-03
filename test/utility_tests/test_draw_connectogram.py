import unittest

import networkx as nx
import numpy as np
from photonai_graph.GraphUtilities import draw_connectogram
from photonai_graph.GraphConversions import dense_to_networkx


class DrawConnectogramTests(unittest.TestCase):

    def setUp(self):
        self.cyc_graph = nx.cycle_graph(20)
        self.weight_graph = dense_to_networkx(np.random.rand(1, 20, 20, 2), adjacency_axis=0)

    def test_drawing(self):
        draw_connectogram(self.cyc_graph, show=False)

    def test_drawing_node_shape(self):
        draw_connectogram(self.cyc_graph, node_shape='x', show=False)

    def test_drawing_colorscheme(self):
        draw_connectogram(self.cyc_graph, colorscheme="Blues", show=False)

    def test_drawing_nodesize(self):
        draw_connectogram(self.cyc_graph, nodesize=200, show=False)

    def test_drawing_weight(self):
        with self.assertRaises(KeyError):
            draw_connectogram(self.cyc_graph, weight=0.5, show=False)

    def test_drawing_weight_corrected(self):
        draw_connectogram(self.weight_graph[0], weight=0.5, show=False)
