import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_knn import GraphConstructorKNN


class KNNTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        self.y = np.ones((20))

    def test_knn_mean(self):
        g_constr = GraphConstructorKNN(transform_style="mean")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_knn_individual(self):
        g_constr = GraphConstructorKNN(transform_style="individual")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_knn_assert_failure(self):
        with self.assertRaises(AssertionError):
            g_constr = GraphConstructorKNN(transform_style="individual")
            g_constr.fit(self.X4d, self.y)
            trans = g_constr.transform(self.X4d)
