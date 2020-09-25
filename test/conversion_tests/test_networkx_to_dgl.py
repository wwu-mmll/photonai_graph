import unittest
import networkx as nx
import numpy as np
import dgl
from photonai_graph.GraphConversions import networkx_to_dgl


class NetworkxToDglTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.cycle_graph(20)] * 10
        self.check_graph = nx.cycle_graph(5)
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_list(self):
        mtrx = networkx_to_dgl(self.graphs)
        self.assertEqual(type(mtrx), list)

    def test_dgl_type(self):
        mtrx = networkx_to_dgl(self.graphs)
        self.assertEqual(type(mtrx[0]), dgl.DGLGraph)

    def test_single_graph(self):
        mtrx = networkx_to_dgl(self.graph)
        self.assertEqual(type(mtrx), dgl.DGLGraph)

    def test_num_nodes(self):
        mtrx = networkx_to_dgl(self.check_graph)
        self.assertEqual(mtrx.number_of_nodes(), 5)

    def test_num_edges(self):
        mtrx = networkx_to_dgl(self.check_graph)
        self.assertEqual(mtrx.number_of_edges(), 10)

    def test_check_nonsense_input(self):
        with self.assertRaises(ValueError):
            mtrx = networkx_to_dgl(self.edge_dict)
