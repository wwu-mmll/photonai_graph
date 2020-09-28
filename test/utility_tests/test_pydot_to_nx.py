import unittest
import networkx as nx
from photonai_graph.GraphUtilities import pydot_to_nx


class PydotToNxTest(unittest.TestCase):

    def setUp(self):
        nx_graph = nx.erdos_renyi_graph(20, 0.3)
        py_graph = nx.nx_pydot.to_pydot(nx_graph)
        self.nx_graph = nx_graph
        self.graph_single = py_graph
        self.graph_list = [py_graph] * 3

    def test_reading_single(self):
        nx_graph = pydot_to_nx(self.graph_single)
        self.assertEqual(type(nx_graph), nx.classes.Graph)

    def test_reading_list(self):
        nx_graphs = pydot_to_nx(self.graph_list)
        self.assertEqual(type(nx_graphs), list)

    def test_reading_list_type(self):
        nx_graphs = pydot_to_nx(self.graph_list)
        self.assertEqual(type(nx_graphs[0]), nx.classes.Graph)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            pydot_to_nx(self.nx_graph)
