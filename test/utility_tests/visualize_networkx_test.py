import unittest
import numpy as np
import networkx as nx
from photonai_graph.GraphUtilities import visualize_networkx


class VisualizeNetworkxTest(unittest.TestCase):

    def setUp(self):
        self.matrix = np.random.rand(20, 20)
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.erdos_renyi_graph(20, 0.3)] * 20

    def test_plot_single(self):
        visualize_networkx(self.graph)

    def test_plot_list(self):
        visualize_networkx(self.graphs)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            visualize_networkx(self.matrix)


if __name__ == '__main__':
    unittest.main()
