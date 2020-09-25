import unittest
import numpy as np
import networkx as nx
import os
from photonai_graph.GraphMeasureTransform import GraphMeasureTransform


class GraphMeasureTransformTests(unittest.TestCase):

    def setUp(self):
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 10
        self.y = np.random.rand(10)
        self.ids = list(range(10))

    def test_transform_default_measures(self):
        g_transform = GraphMeasureTransform()
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_own_measures(self):
        g_transform = GraphMeasureTransform(graph_functions={"global_efficiency": {},
                                                             "local_efficiency": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_list(self):
        g_transform = GraphMeasureTransform(graph_functions={"voterank": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_dict_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"communicability": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_float_or_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"clustering": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

    def test_transform_tuple_dict(self):
        g_transform = GraphMeasureTransform(graph_functions={"hits": {}})
        g_transform.fit(self.X_nx, self.y)
        measures = g_transform.transform(self.X_nx)
        print(measures.shape)

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

    def test_extract_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"eigenvector_centrality": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_list(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"voterank": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)

    def test_extract_dict_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"communicability": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)

    def test_extract_float_or_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"clustering": {}})
        g_transform.extract_measures(self.X_nx, path, self.ids)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_extract_tuple_dict(self):
        path = "/tmp/test.csv"
        g_transform = GraphMeasureTransform(graph_functions={"hits": {}})
        with self.assertRaises(NotImplementedError):
            g_transform.extract_measures(self.X_nx, path, self.ids)


if __name__ == '__main__':
    unittest.main()
