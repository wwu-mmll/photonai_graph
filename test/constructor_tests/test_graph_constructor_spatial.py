import unittest
import os
import numpy as np
import pandas as pd
from photonai_graph.GraphConstruction.graph_constructor_spatial import GraphConstructorSpatial
from photonai_graph.GraphConversions import dense_to_networkx
from photonai_graph.GraphUtilities import visualize_networkx


class KNNTests(unittest.TestCase):

    def setUp(self):
        # get random matrix as feature matrix
        self.X4d = np.ones((20, 12, 12, 2))
        self.Xrandom4d = np.random.rand(20, 12, 12, 2)
        self.y = np.ones((20))

        # get test atlas
        coord_dict = {'x': [39, 42, 42, 45, 3, 6, 6, 9, 42, 39, 45, 42],
                      'y': [45, 39, 42, 42, 3, 3, 3, 3, 3, 3, 3, 3],
                      'z': [45, 45, 39, 42, 42, 39, 45, 48, 3, 6, 6, 9]}
        self.spatial_coords = pd.DataFrame(data=coord_dict)

        # get check array
        arr = np.array([[3]])
        self.check_array = np.tile(arr, (1, 12))

    def test_spatial(self):
        path = "/tmp/test_coords.csv"
        self.spatial_coords.to_csv(path, header=False, index=False)
        g_constr = GraphConstructorSpatial(k_distance=3, atlas_name='test', atlas_folder="/tmp/")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform_test(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 12, 12, 3))
        os.remove(path)

    def test_spatial_mechanism(self):
        path = "/tmp/test_coords.csv"
        self.spatial_coords.to_csv(path, header=False, index=False)
        g_constr = GraphConstructorSpatial(k_distance=3, atlas_name='test', atlas_folder="/tmp/")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform_test(self.Xrandom4d)
        bool_array = np.count_nonzero(trans[0, :, :, 0], axis=1, keepdims=True) == self.check_array
        self.assertFalse(False in bool_array)
        os.remove(path)
