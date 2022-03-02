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
        # create 1d array of networkx graphs
        self.nx_array = np.array([nx.cycle_graph(20), nx.cycle_graph(15), nx.cycle_graph(20)])

    def test_list(self):
        mtrxs = networkx_to_dgl(self.graphs)
        self.assertEqual(type(mtrxs), list)
        for mtrx in mtrxs:
            self.assertEqual(type(mtrx), dgl.DGLGraph)

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
            networkx_to_dgl(self.edge_dict)

    def test_check_nonsense_list_input(self):
        with self.assertRaises(ValueError):
            networkx_to_dgl([self.edge_dict] * 10)

    def test_check_nonsense_array_input(self):
        with self.assertRaises(ValueError):
            networkx_to_dgl(np.array([self.edge_dict] * 10))

    def test_nx_array(self):
        mtrxs = networkx_to_dgl(self.nx_array)
        for mtrx in mtrxs:
            self.assertEqual(type(mtrx), dgl.DGLGraph)
