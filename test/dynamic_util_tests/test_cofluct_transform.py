import unittest
import numpy as np
from photonai_graph.DynamicUtils.cofluct_transform import CofluctTransform


class CofluctTransformTests(unittest.TestCase):

    def setUp(self):
        self.X_test2d = np.random.rand(20, 2000)
        self.X_test3d = np.random.rand(20, 20, 2000)
        self.X_test4d = np.random.rand(20, 20, 2000, 2)
        self.y = np.random.rand(20)

    def test_3d(self):
        coflu = CofluctTransform(quantiles=(0, 0.1), return_mat=True)
        coflu.fit(self.X_test3d, self.y)
        vec = coflu.transform(self.X_test3d)
        self.assertEqual(np.shape(vec), (20, 20, 20))

    def test_4d(self):
        coflu = CofluctTransform(quantiles=(0, 1), return_mat=True)
        coflu.fit(self.X_test4d, self.y)
        vec = coflu.transform(self.X_test4d)
        self.assertEqual(np.shape(vec), (20, 20, 20))

    def test_2d_error(self):
        coflu = CofluctTransform(quantiles=(0, 1), return_mat=True)
        with self.assertRaises(TypeError):
            coflu.fit(self.X_test2d, self.y)
            coflu.transform(self.X_test2d)

    def test_return_dict(self):
        coflu = CofluctTransform(quantiles=(0, 1), return_mat=False)
        coflu.fit(self.X_test3d, self.y)
        vec = coflu.transform(self.X_test3d)
        self.assertEqual(type(vec[0]), dict)
