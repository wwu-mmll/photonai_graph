import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_knn import GraphConstructorKNN


class KNNTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        test_array4d = np.array([[1, 1, 1, 0.3, 0.2, 0.4],
                                 [1, 1, 1, 0.2, 0.1, 0.1],
                                 [1, 1, 1, 0.4, 0.2, 0.2],
                                 [0.2, 0.3, 0.1, 1, 1, 1],
                                 [0.3, 0.4, 0.3, 1, 1, 1],
                                 [0.1, 0.4, 0.3, 1, 1, 1]])
        test_array4d = test_array4d[np.newaxis, :, :, np.newaxis]
        self.Xtest4d = np.repeat(test_array4d, 10, axis=0)
        self.y = np.ones((20))

    def test_knn_individual(self):
        g_constr = GraphConstructorKNN()
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 20, 20, 3))

    def test_knn_assert_failure(self):
        with self.assertRaises(AssertionError):
            g_constr = GraphConstructorKNN()
            g_constr.fit(self.X4d, self.y)
            g_constr.transform(self.X4d)

    def test_knn_mechanism(self):
        g_constr = GraphConstructorKNN(k_distance=2)
        g_constr.fit(self.Xtest4d, self.y)
        trans = g_constr.transform(self.Xtest4d)
        bool_mask = [[True, False, False, True, True, True],
                     [False, True, False, True, True, True],
                     [False, False, True, True, True, True],
                     [True, True, True, True, False, False],
                     [True, True, True, False, True, False],
                     [True, True, True, False, False, True]]
        bool_array = trans[0, :, :, 0] == 0
        comp = bool_mask == bool_array
        comp = comp.all()
        self.assertTrue(comp)
