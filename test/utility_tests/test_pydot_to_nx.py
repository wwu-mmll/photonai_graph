import unittest
import networkx as nx
import os
from photonai_graph.GraphUtilities import pydot_to_nx


class PydotToNxTest(unittest.TestCase):

    def setUp(self):
        self.graphs = [nx.erdos_renyi_graph(20, 0.3)] * 20
