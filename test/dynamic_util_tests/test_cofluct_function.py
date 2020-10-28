import unittest
import numpy as np
from photonai_graph.DynamicUtils.cofluct_functions import cofluct


class CofluctFunctionTests(unittest.TestCase):

    def setUp(self):
        self.X_test2d = np.random.rand(20, 2000)

    def test_cofluct(self):
        res = cofluct(self.X_test2d, (0, 1), return_mat=True)
        self.assertEqual(np.testing.assert_array_almost_equal(res, np.corrcoef(self.X_test2d)), None)

    def test_dict(self):
        res = cofluct(self.X_test2d, (0, 1), return_mat=False)
        self.assertEqual(len(res), 191)
