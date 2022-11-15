import unittest
from tempfile import mkstemp

import numpy as np
import igraph
import os
from photonai_graph.Measures.IgraphMeasureTransform import IgraphMeasureTransform


class GraphMeasureTransformTests(unittest.TestCase):

    def setUp(self):
        self.er_graph = [igraph.Graph.Erdos_Renyi(n=10, p=0.5)] * 5
        self.y = [1, 1, 1, 0, 0]
        self.ids = [1, 2, 3, 4, 5]
        # generate nonsense input
        self.edge_dict = {(1, 0): 1, (2, 0): 1, (3, 0): 1}
        self.string_list = ['test'] * 5

    def test_default_measures(self):
        g_transform = IgraphMeasureTransform()
        g_transform.fit(self.er_graph, self.y)
        measures = g_transform.transform(self.er_graph)
        self.assertEqual(measures.shape, (5, 20))

    def test_transform_nonsense_input(self):
        g_transform = IgraphMeasureTransform()
        with self.assertRaises(TypeError):
            g_transform.fit(self.edge_dict, self.y)
            g_transform.transform(self.edge_dict)

    def test_transform_nonsense_list(self):
        g_tranform = IgraphMeasureTransform()
        with self.assertRaises(TypeError):
            g_tranform.transform(self.string_list)

    def test_transform_own_measures_parallel(self):
        g_transform = IgraphMeasureTransform(n_processes=2)
        g_transform.fit(self.er_graph, self.y)
        measures = g_transform.transform(self.er_graph)
        self.assertEqual(measures.shape, (5, 20))

    def test_transform_check_output_type(self):
        g_transform = IgraphMeasureTransform()
        g_transform.fit(self.er_graph, self.y)
        measures = g_transform.transform(self.er_graph)
        self.assertEqual(type(measures), np.ndarray)

    def test_transform_measure_single(self):
        g_transform = IgraphMeasureTransform(graph_functions={"radius": {}})
        g_transform.fit(self.er_graph, self.y)
        measures = g_transform.transform(self.er_graph)
        self.assertEqual(measures.shape, (5, 1))

    def test_transform_measure_multiple(self):
        g_transform = IgraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.fit(self.er_graph, self.y)
        measures = g_transform.transform(self.er_graph)
        self.assertEqual(measures.shape, (5, 10))

    def test_extract_nonsense_input_no_ids(self):
        g_transform = IgraphMeasureTransform()
        with self.assertRaises(ValueError):
            g_transform.fit(self.edge_dict, self.y)
            g_transform.extract_measures(self.edge_dict)

    def test_extract_default_measures(self):
        _, path = mkstemp('.csv')
        g_transform = IgraphMeasureTransform()
        g_transform.extract_measures(self.er_graph, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_own_measures(self):
        _, path = mkstemp('.csv')
        g_transform = IgraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.extract_measures(self.er_graph, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
