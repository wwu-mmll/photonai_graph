import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_random_walks import GraphConstructorRandomWalks


class RWTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        self.y = np.ones((20))

    def test_rw_mean(self):
        g_constr = GraphConstructorRandomWalks()
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_rw_individual(self):
        g_constr = GraphConstructorRandomWalks()
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_rw_assert_failure(self):
        with self.assertRaises(AssertionError):
            g_constr = GraphConstructorRandomWalks()
            g_constr.fit(self.X4d, self.y)
            g_constr.transform(self.X4d)
