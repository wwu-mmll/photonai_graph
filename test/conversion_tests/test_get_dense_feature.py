import unittest
import numpy as np
from photonai_graph.GraphConversions import get_dense_feature


class GetDenseFeatureTest(unittest.TestCase):

    def setUp(self):
        id_matrix = np.eye(5)
        id_array = id_matrix[:, :, np.newaxis]
        self.check_matrix = np.repeat(id_array, 2, axis=2)
        self.control_dict = {0: [1, 0, 0, 0, 0],
                             1: [0, 1, 0, 0, 0],
                             2: [0, 0, 1, 0, 0],
                             3: [0, 0, 0, 1, 0],
                             4: [0, 0, 0, 0, 1]}

    def test_return_type(self):
        feat = get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(feat), dict)

    def test_sum(self):
        feat = get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1,
                                 aggregation="sum")
        self.assertTrue(all(value == 1 for value in feat.values()))

    def test_mean(self):
        feat = get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1,
                                 aggregation="mean")
        self.assertTrue(all(value == 0.2 for value in feat.values()))

    def test_node_degree(self):
        feat = get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1,
                                 aggregation="node_degree")
        self.assertTrue(all(value == 1 for value in feat.values()))

    def test_features(self):
        feat = get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1,
                                 aggregation="features")
        self.assertTrue(feat == self.control_dict)

    def test_error(self):
        with self.assertRaises(KeyError):
            get_dense_feature(self.check_matrix, adjacency_axis=0, feature_axis=1,
                              aggregation="weight")
