import unittest
import numpy as np
from photonai_graph.GraphConversions import get_dense_edge_features


class GetDenseEdgeFeaturesTest(unittest.TestCase):

    def setUp(self):
        id_matrix = np.eye(3)
        id_array = id_matrix[:, :, np.newaxis]
        self.check_matrix = np.repeat(id_array, 2, axis=2)
        self.control_dict = {('0', '0'): 1.0,
                             ('0', '1'): 0.0,
                             ('0', '2'): 0.0,
                             ('1', '0'): 0.0,
                             ('1', '1'): 1.0,
                             ('1', '2'): 0.0,
                             ('2', '0'): 0.0,
                             ('2', '1'): 0.0,
                             ('2', '2'): 1.0}

    def test_return_type(self):
        feat = get_dense_edge_features(self.check_matrix, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(feat), dict)

    def test_dict_content(self):
        feat = get_dense_edge_features(self.check_matrix, adjacency_axis=0, feature_axis=1)
        self.assertTrue(feat == self.control_dict)
