import unittest
import numpy as np
from photonai_graph.GraphConstruction.graph_constructor_threshold import GraphConstructorThreshold


class ThresholdTests(unittest.TestCase):

    def setUp(self):
        self.X4d_adjacency = np.ones((20, 20, 20, 1))
        self.X4d_features = np.random.rand(20, 20, 20, 1)
        self.X4d = np.concatenate((self.X4d_adjacency, self.X4d_features), axis=3)
        self.X3d = np.ones((20, 20, 20))
        self.y = np.ones(20)

    def test_threshold_3d(self):
        # ensure that individual transform style with a 3d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5)
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))
        # thresholding an array of ones with 0.5 should not change anything
        self.assertTrue(np.array_equal(trans[..., 0], self.X3d))
        # as the original connectivity is not discarded, it should be present in second dim
        self.assertTrue(np.array_equal(trans[..., 1], self.X3d))

    def test_threshold_3d_discard_connectivity(self):
        g_constr = GraphConstructorThreshold(threshold=0.5,
                                             discard_original_connectivity=True)
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 1))
        self.assertTrue(np.array_equal(trans[..., 0], self.X3d))

    def test_threshold_4d(self):
        # ensure that individual transform style with a 4d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 3))
        # first dimension should be thresholded but unchanged
        self.assertTrue(np.array_equal(trans[..., 0, np.newaxis], self.X4d_adjacency))
        # second dimension should contain original connectivity
        self.assertTrue(np.array_equal(trans[..., 1, np.newaxis], self.X4d_adjacency))
        # last dimension should contain the random features
        self.assertTrue(np.array_equal(trans[..., 2, np.newaxis], self.X4d_features))

    def test_threshold_4d_discard_connectivity(self):
        # ensure that individual transform style with a 4d matrix returns the right shape
        g_constr = GraphConstructorThreshold(threshold=0.5,
                                             discard_original_connectivity=True)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 2))
        # first dimension should be thresholded but unchanged
        self.assertTrue(np.array_equal(trans[..., 0, np.newaxis], self.X4d_adjacency))
        # last dimension should contain the random features
        self.assertTrue(np.array_equal(trans[..., 1, np.newaxis], self.X4d_features))

    def test_threshold_shape_3d_onehot(self):
        # ensure that an individual transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, one_hot_nodes=1)
        g_constr.fit(self.X3d, self.y)
        trans = g_constr.transform(self.X3d)
        self.assertEqual(trans.shape, (20, 20, 20, 3))
        # the first dimension contains the original connectivity
        self.assertTrue(np.array_equal(trans[..., 0], self.X3d))
        # the second dimension contains the one hot encoding
        # we know the one hot encoding, as we created the matrix accordingly
        self.assertTrue(np.array_equal(trans[..., 1, np.newaxis],
                                       np.repeat(np.eye(20)[np.newaxis, ...], 20, axis=0)[..., np.newaxis]))
        # the last dimension contains the original connectivity again
        self.assertTrue(np.array_equal(trans[..., 2], self.X3d))

    def test_threshold_shape_4d_onehot(self):
        # ensure that an individual transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5, one_hot_nodes=1)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 4))
        # the first dimension still contains the (thresholded but unchanged) values
        self.assertTrue(np.array_equal(trans[..., 0, np.newaxis], self.X4d_adjacency))
        # the second dimesion contains the one hot encoding
        # We know the one hot encoding, as we created the matrix accordingly
        self.assertTrue(np.array_equal(trans[..., 1, np.newaxis],
                                       np.repeat(np.eye(20)[np.newaxis, ...], 20, axis=0)[..., np.newaxis]))
        # the third dimension contains again the original values
        self.assertTrue(np.array_equal(trans[..., 2, np.newaxis], self.X4d_adjacency))
        # the last dimension contains the features
        self.assertTrue(np.array_equal(trans[..., 3, np.newaxis], self.X4d_features))

    def test_threshold_individual_shape_4d_onehot_discard_connectivity(self):
        # ensure that an individual transform with a 3d matrix returns the right shape
        # when using one hot encoded features
        g_constr = GraphConstructorThreshold(threshold=0.5,
                                             one_hot_nodes=1,
                                             discard_original_connectivity=True)
        g_constr.fit(self.X4d, self.y)
        trans = g_constr.transform(self.X4d)
        self.assertEqual(trans.shape, (20, 20, 20, 3))
        # the first dimension still contains the (thresholded but unchanged) values
        self.assertTrue(np.array_equal(trans[..., 0, np.newaxis], self.X4d_adjacency))
        # the second dimesion contains the one hot encoding
        # We know the one hot encoding, as we created the matrix accordingly
        self.assertTrue(np.array_equal(trans[..., 1, np.newaxis],
                                       np.repeat(np.eye(20)[np.newaxis, ...], 20, axis=0)[..., np.newaxis]))
        # the last dimension contains the features
        self.assertTrue(np.array_equal(trans[..., 2, np.newaxis], self.X4d_features))
