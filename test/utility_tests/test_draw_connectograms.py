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
        self.directory = './'

    def test_drawing_single(self):
        draw_connectograms(self.cyc_graph, show=False)

    def test_drawing_list(self):
        draw_connectograms(self.cyc_graphs, show=False)

    def test_drawing_len_ids(self):
        output_format = ".svg"
        draw_connectograms(self.cyc_graphs, path=self.directory,
                           ids=self.ids, out_format=output_format, show=False)
        self.assertTrue(os.path.exists(self.directory + str(6) + output_format))
        [os.remove(os.path.join(self.directory, f"{cid}{output_format}")) for cid in self.ids]

    def test_drawing_len_ids_wrong(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, path=self.directory,
                               ids=self.ids_wrong, out_format='.svg', show=False)

    def test_drawing_path(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, ids=self.ids, out_format='.svg', show=False)

    def test_drawing_outformat(self):
        with self.assertRaises(Exception):
            draw_connectograms(self.cyc_graphs, path=self.directory,
                               ids=self.ids, show=False)

    def test_save_no_ids(self):
        output_format = ".png"
        draw_connectograms(self.cyc_graphs, path=self.directory, out_format=output_format, show=False)
        self.assertTrue(os.path.exists(self.directory + str(0) + output_format))
        [os.remove(os.path.join(self.directory, f"{cid}{output_format}")) for cid, _ in enumerate(self.cyc_graphs)]

    def test_format(self):
        with self.assertRaises(TypeError):
            draw_connectograms(self.matrix, show=False)
