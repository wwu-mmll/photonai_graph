import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_threshold import GraphConstructorThreshold


class ThresholdTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.X3d = np.ones((20, 20, 20))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        self.y = np.ones((20))

    def test_threshold_individual_3d(self):
        # ensure that a individual transform style on a 3d matrix does not return values other than one
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual")
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(np.min(trans), 1)
        self.assertEqual(np.max(trans), 1)

    def test_threshold_individual_4d(self):
        # ensure that a individual transform style on a 4d matrix does not return values other than one
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual")
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(np.min(trans), 1)
        self.assertEqual(np.max(trans), 1)

    def test_threshold_mean_3d(self):
        # ensure that a individual transform style on a 3d matrix does not return values other than zero
        g_constr = GraphConstructorThreshold(threshold=1.5, transform_style="individual")
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(np.min(trans[:, :, :, 0]), 0)
        self.assertEqual(np.max(trans[:, :, :, 0]), 0)

    def test_threshold_mean_4d(self):
        # ensure that a individual transform style on a 4d matrix does not return values other than zero
        g_constr = GraphConstructorThreshold(threshold=1.5, transform_style="individual")
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(np.min(trans[:, :, :, 0]), 0)
        self.assertEqual(np.max(trans[:, :, :, 0]), 0)

    def test_threshold_retain_weights(self):
        # ensure that retain weights leads to a diverse set of weights (more than 0 and 1)
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual", retain_weights=1)
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.min(trans[:, :, :, 0]), 0)
        self.assertGreaterEqual(len(np.unique(trans[:, :, :, 0])), 2)

    def test_threshold_retain_weights_error(self):
        # ensure a Value error is being raised when retain weights is neither 0 or 1
        with self.assertRaises(ValueError):
            g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual", retain_weights=0.5)
            g_constr.fit(self.Xrandom4d, self.y)
            trans = g_constr.transform(self.Xrandom4d)


if __name__ == '__main__':
    unittest.main()
