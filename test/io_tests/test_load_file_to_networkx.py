import unittest
import warnings

import networkx as nx
import numpy as np

from photonai_graph.GraphIO import save_networkx_to_file, load_file_to_networkx


class LoadFileTNetworkxTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)

    def test_input_format_wrong(self):
         in_format = "heavy_weight_edge_list"
         path = "/spm-data/Scratch/spielwiese_vincent/tmp/"
         with self.assertRaises(KeyError):
            load_file_to_networkx(path, input_format=in_format)

    def test_load_list(self):
        # in_format = "dot"
        # path = ["/spm-data/Scratch/spielwiese_vincent/tmp/graph_1",
        #        "/spm-data/Scratch/spielwiese_vincent/tmp/graph_2",
        #        "/spm-data/Scratch/spielwiese_vincent/tmp/graph_3"]
        # graphs = load_file_to_networkx(path, in_format)
        # self.assertEqual(type(graphs), list)
        warnings.warn('This test has to be reimplemented')

    def test_load_graph(self):
        # in_format = "dot"
        # path = ["/spm-data/Scratch/spielwiese_vincent/tmp/graph_1",
        #        "/spm-data/Scratch/spielwiese_vincent/tmp/graph_2",
        #        "/spm-data/Scratch/spielwiese_vincent/tmp/graph_3"]
        # graphs = load_file_to_networkx(path, in_format)
        # self.assertEqual(type(graphs[0]), nx.classes.MultiGraph)
        warnings.warn('This test has to be reimplemented')
