import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_percentage_window import GraphConstructorPercentageWindow


class PercentageWindowTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        test_array = np.reshape(np.arange(1, 101, 1), (-1, 10, 10))
        self.Xtest4d = np.repeat(test_array, 10, axis=0)[..., np.newaxis]
        self.y = np.ones((20))

    def test_percentage_window_individual(self):
        g_constr = GraphConstructorPercentageWindow()
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_percentage_window_retain_weights_none(self):
        g_constr = GraphConstructorPercentageWindow(retain_weights=0)
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(len(np.unique(trans[:, :, :, 0])), 2)

    def test_percentage_retain_weights(self):
        # ensure that retain weights leads to a diverse set of weights (more than 0 and 1)
        g_constr = GraphConstructorPercentageWindow(retain_weights=1)
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.min(trans[:, :, :, 0]), 0)
        self.assertGreaterEqual(len(np.unique(trans[:, :, :, 0])), 2)

    def test_threshold_window_retain_weights_error(self):
        # ensure a Value error is being raised when retain weights is neither 0 or 1
        with self.assertRaises(ValueError):
            g_constr = GraphConstructorPercentageWindow(retain_weights=0.5)
            g_constr.fit(self.Xrandom4d, self.y)
            g_constr.transform(self.Xrandom4d)

    def test_percentage_contains(self):
        # ensure that the threshold actually picks the right rows of the matrix
        g_constr = GraphConstructorPercentageWindow(percentage_upper=80,
                                                    percentage_lower=60,
                                                    retain_weights=1)
        g_constr.fit(self.Xtest4d, self.y)
        trans = g_constr.transform(self.Xtest4d)
        expected_elements = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        for element in expected_elements:
            self.assertIn(element, trans[0, :, :, 0])

    def test_percentage_not_contains(self):
        # ensure that the threshold actually picks the right rows of the matrix
        g_constr = GraphConstructorPercentageWindow(percentage_upper=80,
                                                    percentage_lower=60,
                                                    retain_weights=1)
        g_constr.fit(self.Xtest4d, self.y)
        trans = g_constr.transform(self.Xtest4d)
        expected_elements = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        for element in expected_elements:
            self.assertNotIn(element, trans[0, :, :, 0])
