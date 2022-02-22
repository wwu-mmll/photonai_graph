import unittest
import numpy as np
from photonai_graph.GraphUtilities import get_random_labels


class DrawConnectogramTests(unittest.TestCase):

    def setUp(self):
        self.np_control = np.random.rand(20)

    def test_type(self):
        y = get_random_labels(number_of_labels=20)
        self.assertEqual(type(y), np.ndarray)

    def test_shape(self):
        y = get_random_labels(number_of_labels=20)
        self.assertEqual(np.shape(y), (20,))

    def test_classification(self):
        y = get_random_labels(number_of_labels=20)
        self.assertEqual(len(np.unique(y)), 2)

    def test_regression(self):
        y = get_random_labels(l_type="regression", number_of_labels=20)
        self.assertGreaterEqual(len(np.unique(y)), 3)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            get_random_labels(l_type="prediction", number_of_labels=20)
