import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_threshold import GraphConstructorThreshold


class ThresholdTests(unittest.TestCase):

    def setUp(self):
        self.X4d = np.ones((20, 20, 20, 2))
        self.X3d = np.ones((20, 20, 20))
        self.Xrandom4d = np.random.rand(20, 20, 20, 2)
        self.y = np.ones((20))

    def test_threshold_mean_shape_3d(self):
        # ensure that mean transform style with a 3d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="mean")
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))

    def test_threshold_mean_shape_4d(self):
        # ensure that mean transform style with a 4d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="mean")
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 3))

    def test_threshold_individual_shape_3d(self):
        # ensure that individual transform style with a 3d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual")
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))

    def test_threshold_individual_shape_4d(self):
        # ensure that individual transform style with a 4d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual")
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 3))

    def test_threshold_mean_shape_3d_onehot(self):
        # ensure that a mean transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="mean", one_hot_nodes=1)
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))

    def test_threshold_mean_shape_4d_onehot(self):
        # ensure that a mean transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="mean", one_hot_nodes=1)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))

    def test_threshold_individual_shape_3d_onehot(self):
        # ensure that an individual transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual", one_hot_nodes=1)
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))

    def test_threshold_individual_shape_4d_onehot(self):
        # ensure that an individual transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, transform_style="individual", one_hot_nodes=1)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))
