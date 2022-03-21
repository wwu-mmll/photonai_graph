import unittest
import os
import numpy as np
import pandas as pd
from photonai_graph.GraphConstruction.graph_constructor_spatial import GraphConstructorSpatial


class SpatialTests(unittest.TestCase):

    def setUp(self):
        # get random matrix as feature matrix
        self.X4d = np.ones((20, 12, 12, 2))
        self.Xrandom4d = np.random.rand(20, 12, 12, 2)
        self.XrandomHO4d = np.random.rand(20, 110, 110, 2)
        self.y = np.ones((20))

        # get test atlas
        coord_dict = {'x': [39, 42, 42, 45, 3, 6, 6, 9, 42, 39, 45, 42],
                      'y': [45, 39, 42, 42, 3, 3, 3, 3, 3, 3, 3, 3],
                      'z': [45, 45, 39, 42, 42, 39, 45, 48, 3, 6, 6, 9]}
        self.spatial_coords = pd.DataFrame(data=coord_dict)

        # get check array
        arr = np.array([[3]])
        self.check_array = np.tile(arr, (1, 12))

        # get check array for ho_atlas
        self.ho_check = [4, 3, 4, 5, 5, 4, 3, 4, 3, 4, 5,
                         4, 4, 5, 3, 3, 5, 5, 3, 4, 3, 3,
                         4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4,
                         3, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3,
                         3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 4,
                         4, 4, 4, 3, 3, 5, 5, 3, 4, 4, 3,
                         6, 6, 4, 4, 4, 4, 3, 3, 5, 6, 6,
                         7, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3,
                         4, 3, 4, 3, 4, 3, 4, 4, 4, 5, 4,
                         5, 5, 4, 4, 4, 4, 4, 6, 4, 3, 3]

    def test_spatial(self):
        path = "/tmp/test_coords.csv"
        self.spatial_coords.to_csv(path, header=False, index=False)
        g_constr = GraphConstructorSpatial(k_distance=3, atlas_name='test', atlas_folder="/tmp/")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        self.assertEqual(np.shape(trans), (20, 12, 12, 3))
        os.remove(path)

    def test_spatial_mechanism(self):
        path = "/tmp/test_coords.csv"
        self.spatial_coords.to_csv(path, header=False, index=False)
        g_constr = GraphConstructorSpatial(k_distance=3, atlas_name='test', atlas_folder="/tmp/")
        g_constr.fit(self.Xrandom4d, self.y)
        trans = g_constr.transform(self.Xrandom4d)
        bool_array = np.count_nonzero(trans[0, :, :, 0], axis=1, keepdims=True) == self.check_array
        self.assertFalse(False in bool_array)
        os.remove(path)

    def test_ho_atlas(self):
        path = os.path.dirname(os.path.abspath(__file__))
        g_constr = GraphConstructorSpatial(k_distance=3, atlas_name='ho', atlas_folder=path)
        g_constr.fit(self.XrandomHO4d, self.y)
        trans = g_constr.transform(self.XrandomHO4d)
        ho_adj = trans[0, :, :, 0]
        self.assertTrue(np.array_equal(np.count_nonzero(ho_adj, axis=0), self.ho_check))
