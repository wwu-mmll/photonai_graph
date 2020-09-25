import unittest
import numpy as np
import networkx as nx
from photonai_graph.GraphUtilities import check_asteroidal


class CheckAsteroidalTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.cycle_graph(20)] * 10
        self.nonsense = np.random.rand(20, 20)

    def test_output_bool(self):
        g = check_asteroidal(self.graph, return_boolean=True)
        self.assertFalse(g)

    def test_output_nobool(self):
        g = check_asteroidal(self.graph, return_boolean=False)
        self.assertEqual(g, [6, 9, 0])

    def test_output_list_bool(self):
        g = check_asteroidal(self.graphs, return_boolean=True)
        f_list = [False] * 10
        self.assertEqual(g, f_list)

    def test_output_list_nobool(self):
        g = check_asteroidal(self.graphs, return_boolean=False)
        v_list = [[6, 9, 0]] * 10
        self.assertEqual(g, v_list)

    def test_nonsense_bool(self):
        with self.assertRaises(ValueError):
            g = check_asteroidal(self.nonsense, return_boolean=True)

    def test_nonsense_nobool(self):
        with self.assertRaises(ValueError):
            g = check_asteroidal(self.nonsense, return_boolean=False)
