import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_threshold_window import GraphConstructorThresholdWindow


class ThresholdWindowTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.X3d = np.ones((20, 20, 20))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        self.y = np.ones((20))

    def test_treshold_window_mean(self):
        g_constr = GraphConstructorThresholdWindow()
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_threshold_window_individual(self):
        g_constr = GraphConstructorThresholdWindow(transform_style="individual")
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_threshold_window_retain_weights_none(self):
        g_constr = GraphConstructorThresholdWindow(retain_weights=0)
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(len(np.unique(trans[:, :, :, 0])), 2)

    def test_threshold_window_retain_weights(self):
        # ensure that retain weights leads to a diverse set of weights (more than 0 and 1)
        g_constr = GraphConstructorThresholdWindow(retain_weights=1)
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.min(trans[:, :, :, 0]), 0)
        self.assertGreaterEqual(len(np.unique(trans[:, :, :, 0])), 2)

    def test_threshold_window_retain_weights_error(self):
        # ensure a Value error is being raised when retain weights is neither 0 or 1
        with self.assertRaises(ValueError):
            g_constr = GraphConstructorThresholdWindow(retain_weights=0.5)
            g_constr.fit(self.Xrandom4d, self.y)
            trans = g_constr.transform(self.Xrandom4d)
