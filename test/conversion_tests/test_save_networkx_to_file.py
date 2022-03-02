import unittest
import numpy as np
import networkx as nx
import os
from photonai_graph.GraphConversions import save_networkx_to_file


class SaveNetworkxToFileTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.cycle_graph(20)] * 10
        self.nonsense = np.random.rand(20, 20)
        self.ids = list(range(11, 21))
        self.ids_misfit = list(range(5, 11))

    def test_output_format_wrong(self):
        out_format = "heavy_weight_edge_list"
        path = "/tmp/"
        with self.assertRaises(ValueError):
            save_networkx_to_file(self.graphs, path=path, output_format=out_format)

    def test_ids_length_wrong(self):
        out_format = "dot"
        path = "/tmp/"
        with self.assertRaises(ValueError):
            save_networkx_to_file(self.graphs, path=path, output_format=out_format,
                                  ids=self.ids_misfit)

    def test_no_ids(self):
        out_format = "dot"
        path = "/tmp/"
        save_networkx_to_file(self.graphs, path=path, output_format=out_format)
        self.assertTrue(os.path.exists(path + "graph_0"))

    def test_ids(self):
        out_format = "dot"
        path = "/tmp/"
        save_networkx_to_file(self.graphs, path=path, output_format=out_format, ids=self.ids)
        self.assertTrue(os.path.exists(path + "graph_11"))

    def test_single_graph(self):
        out_format = "dot"
        path = "/tmp/graph25"
        save_networkx_to_file(self.graph, path=path, output_format=out_format)
        self.assertTrue(os.path.exists(path))

    def test_nonsense_input(self):
        out_format = "dot"
        path = "/tmp/graph25"
        with self.assertRaises(ValueError):
            save_networkx_to_file(self.nonsense, path=path, output_format=out_format)

    def test_random_list_input(self):
        out_format = "dot"
        path = "/tmp/graph25"
        with self.assertRaises(ValueError):
            save_networkx_to_file([self.nonsense] * 20, path=path, output_format=out_format)
