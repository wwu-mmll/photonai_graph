import unittest

import numpy as np
import networkx as nx
import os
from photonai_graph.GraphMeasureTransform import GraphMeasureTransform


class GraphMeasureTransformTests(unittest.TestCase):

    def setUp(self):
        gs = np.load(os.path.dirname(__file__) + '/X_test.npz')['arr_0']
        self.X_nx = [nx.from_numpy_array(gs[i]) for i in range(gs.shape[0])]
        self.y = np.random.rand(10)
        self.ids = list(range(10))
        # generate random matrices
        self.random_mtrx = np.random.rand(10, 20, 20, 2)
        # generate nonsense input
        self.edge_dict = {(1, 0): 1, (2, 0): 1, (3, 0): 1}

    def test_transform_input_mtrx(self):
        g_transform = GraphMeasureTransform()
        g_transform.fit(self.random_mtrx, self.y)
        measures = g_transform.transform(self.random_mtrx)
        self.assertEqual(measures.shape, (10, 2))

    def test_transform_nonsense_input(self):
        g_transform = GraphMeasureTransform()
        with self.assertRaises(TypeError):
            g_transform.fit(self.edge_dict, self.y)
            g_transform.transform(self.edge_dict)

    def test_transform_default_measures(self):
        g_transform = GraphMeasureTransform()
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 2))

    def test_transform_own_measures(self):
        g_transform = GraphMeasureTransform(graph_functions={"global_efficiency": {},
                                                             "local_efficiency": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 2))

    def test_transform_own_measures_parallel(self):
        g_transform = GraphMeasureTransform(graph_functions={"global_efficiency": {},
                                                             "local_efficiency": {}},
                                            n_processes=2)
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 2))

    def test_transform_dict_node(self):
        g_transform = GraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 20))

    def test_transform_dict_edge(self):
        g_transform = GraphMeasureTransform(graph_functions={"edge_current_flow_betweenness_centrality": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(type(measures), np.ndarray)

    def test_transform_list(self):
        g_transform = GraphMeasureTransform(graph_functions={"voterank": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(type(measures), np.ndarray)

    def test_transform_dict_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"communicability": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 400))

    def test_transform_float_or_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"clustering": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 20))

    def test_transform_tuple_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"hits": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 40))

    def test_transform_dual_tuple(self):
        g_transform = GraphMeasureTransform(graph_functions={"non_randomness": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 2))

    def test_transform_directed(self):
        g_transform = GraphMeasureTransform(graph_functions={"degree_pearson_correlation_coefficient": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        self.assertEqual(measures.shape, (10, 1))

    def test_extract_input_mtrx(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform()
        g_transform.fit(self.random_mtrx, self.y)
        g_transform.extract_measures(self.random_mtrx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_nonsense_input_no_ids(self):
        g_transform = GraphMeasureTransform()
        with self.assertRaises(ValueError):
            g_transform.fit(self.edge_dict, self.y)
            g_transform.extract_measures(self.edge_dict)

    def test_extract_default_measures(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform()
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_own_measures(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"global_efficiency": {},
                                                             "local_efficiency": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_dict_node(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_dict_edge(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"edge_current_flow_betweenness_centrality": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_float_or_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"clustering": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_no_id(self):
        g_transform = GraphMeasureTransform()
        with self.assertRaises(Exception):
            g_transform.extract_measures(self.X_nx)

    def test_extract_directed(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"degree_pearson_correlation_coefficient": {}})
        g_transform.fit(self.X_nx, self.y)
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_compute_average(self):
        g_transform = GraphMeasureTransform(graph_functions={"degree_centrality": {}})
        g_transform.fit(self.X_nx, self.y)
        X_nodes = g_transform.transform(self.X_nx)
        X_nodes_average = np.mean(X_nodes, axis=1).reshape(-1, 1)

        g_transform = GraphMeasureTransform(graph_functions={"average_degree_centrality": {}})
        g_transform.fit(self.X_nx, self.y)
        X_average = g_transform.transform(self.X_nx)

        np.testing.assert_array_equal(X_average, X_nodes_average)

    # !!!
    #   Below this line are legacy tests where
    #   NotImplementedErrors are expected.
    #   ---------------------------------------
    #   todo: Reimplement all these tests!
    # !!!

    # todo: rewrite
    """def test_extract_list(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"voterank": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)"""

    # todo: rewrite
    """def test_extract_dict_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"communicability": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)"""

    # todo: reimplement
    """def test_extract_tuple_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"hits": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)"""

    # todo: reimplement
    """    
    def test_extract_dual_tuple(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"non_randomness": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)"""
