import unittest
import networkx as nx
import numpy as np
import os
from photonai_graph.GraphUtilities import draw_connectograms


class DrawConnectogramTests(unittest.TestCase):

    def setUp(self):
        self.cyc_graph = nx.cycle_graph(20)
        self.cyc_graphs = [nx.cycle_graph(20)] * 5
        self.matrix = np.random.rand(20, 20, 20)
        self.ids = list(range(6, 11))
        self.ids_wrong = list(range(1, 11))

    def test_drawing_single(self):
        draw_connectograms(self.cyc_graph, show=False)

    def test_drawing_list(self):
        draw_connectograms(self.cyc_graphs, show=False)

    def test_drawing_len_ids(self):
        directory = '/tmp/'
        output_format = ".svg"
        draw_connectograms(self.cyc_graphs, path=directory,
                           ids=self.ids, out_format=output_format, show=False)
        self.assertTrue(os.path.exists(directory + str(6) + output_format))

    def test_drawing_len_ids_wrong(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, path='/tmp/',
                               ids=self.ids_wrong, out_format='.svg', show=False)

    def test_drawing_path(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, ids=self.ids, out_format='.svg', show=False)

    def test_drawing_outformat(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, path='/tmp/',
                               ids=self.ids, show=False)

    def test_save_no_ids(self):
        directory = '/tmp/'
        output_format = ".png"
        draw_connectograms(self.cyc_graphs, path=directory, out_format=output_format, show=False)
        self.assertTrue(os.path.exists(directory + str(0) + output_format))

    def test_format(self):
        with self.assertRaises(TypeError):
            draw_connectograms(self.matrix, show=False)
