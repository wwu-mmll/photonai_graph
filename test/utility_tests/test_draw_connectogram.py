import unittest
import networkx as nx
from photonai_graph.GraphUtilities import draw_connectogram


class DrawConnectogramTests(unittest.TestCase):

    def setUp(self):
        self.cyc_graph = nx.cycle_graph(20)

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
