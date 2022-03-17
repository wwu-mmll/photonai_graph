import unittest
import numpy as np
from photonai_graph.GraphUtilities import individual_ztransform


class IndividualZtransformTest(unittest.TestCase):

    def setUp(self):
        self.m3d = np.random.rand(10, 20, 20)
        self.m4d = np.ones((10, 20, 20, 2))

    def test_ztransform_3d(self):
        with self.assertRaises(ValueError):
            individual_ztransform(self.m3d)

    def test_ztransform_4d(self):
        trans_m4d = individual_ztransform(self.m4d)
        self.assertEqual(np.shape(trans_m4d), np.shape(self.m4d[:, :, :, 0]))
