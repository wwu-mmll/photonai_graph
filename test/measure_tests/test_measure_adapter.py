import unittest

import numpy as np
import networkx as nx
import igraph
from photonai_graph.Measures.MeasureAdapter import GraphMeasureAdapter
from photonai_graph.util import NetworkxGraphWrapper


class MeasureAdapterTests(unittest.TestCase):

    def setUp(self):
        # generate random graphs
        self.random_mtrx = np.random.rand(10, 20, 20, 2)
        self.random_igraph = [igraph.Graph.Erdos_Renyi(n=10, p=0.3) for i in range(10)]
        self.random_networkx = [nx.erdos_renyi_graph(n=10, p=0.3) for i in range(10)]
        # generate nonsense input
        self.edge_dict = {(1, 0): 1, (2, 0): 1, (3, 0): 1}
        self.y = np.random.rand(10)

    def test_np_to_igraph(self):
        adapter = GraphMeasureAdapter(output='igraph')
        adapter.fit(self.random_mtrx, self.y)
        g = adapter.transform(self.random_mtrx)
        self.assertEqual(type(g[0]), igraph.Graph)
        self.assertEqual(type(g), np.ndarray)

    def test_igraph_to_igraph(self):
        adapter = GraphMeasureAdapter(output='igraph')
        g = adapter.transform(self.random_igraph)
        self.assertEqual(type(g[0]), type(self.random_igraph[0]))

    def test_igraph_type_error(self):
        adapter = GraphMeasureAdapter(output='igraph')
        with self.assertRaises(TypeError):
            adapter.transform(self.edge_dict)

    def test_np_to_networkx(self):
        adapter = GraphMeasureAdapter(output='networkx')
        adapter.fit(self.random_mtrx, self.y)
        g = adapter.transform(self.random_mtrx)
        self.assertEqual(type(g[0]), NetworkxGraphWrapper)
        self.assertEqual(type(g), np.ndarray)

    def test_networkx_to_networkx(self):
        adapter = GraphMeasureAdapter(output='networkx')
        g = adapter.transform(self.random_networkx)
        self.assertEqual(type(g[0]), NetworkxGraphWrapper)

    def test_output_type_error(self):
        adapter = GraphMeasureAdapter(output='test')
        with self.assertRaises(NotImplementedError):
            adapter.transform(self.random_mtrx)
