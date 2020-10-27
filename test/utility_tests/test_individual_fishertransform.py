import unittest
import numpy as np
from photonai_graph.GraphUtilities import individual_fishertransform


class IndividualFishertransformTest(unittest.TestCase):

    def setUp(self):
        self.m3d = np.random.rand(10, 20, 20)
        self.m4d = np.ones((10, 20, 20, 2))

    def test_fishertransform_3d(self):
        trans_m3d = individual_fishertransform(self.m3d)
        self.assertEqual(np.shape(trans_m3d), np.shape(self.m3d))

    def test_ztransform_4d(self):
        trans_m4d = individual_fishertransform(self.m4d)
        self.assertEqual(np.shape(trans_m4d), np.shape(self.m4d[:, :, :, 0]))