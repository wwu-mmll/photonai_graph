import unittest
import networkx as nx
import os
from photonai_graph.GraphUtilities import save_graphs


class SaveGraphsTest(unittest.TestCase):

    def setUp(self):
        self.graphs = [nx.erdos_renyi_graph(20, 0.3)] * 20

    def test_save(self):
        directory = "/spm-data/Scratch/spielwiese_vincent/tmp/"
        save_graphs(self.graphs, path=directory)
        self.assertTrue(os.path.exists(directory + "graph_1"), True)

    def test_exception(self):
        with self.assertRaises(NotImplementedError):
            save_graphs(self.graphs, input_format="dense")


if __name__ == '__main__':
    unittest.main()
