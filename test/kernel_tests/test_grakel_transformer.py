import unittest
import numpy as np

from photonai_graph.GraphKernels.grakel_adapter import GrakelAdapter
from photonai_graph.GraphKernels.GrakelTransformer import GrakelTransformer


class GrakelTransformerTest(unittest.TestCase):

    def setUp(self) -> None:
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.Xrandom4d = mtrx
        self.Xrandom_missfit = np.random.rand(20, 20, 5, 2)
        self.y = np.ones(20)

    def test_dummy_function(self):
        g_adapter = GrakelAdapter()
        x_trans = g_adapter.transform(self.Xrandom4d)
        self.assertEqual(type(x_trans), list)
        gt = GrakelTransformer(transformation="ShortestPath", with_labels=True)
        gt.fit(x_trans)
        res = gt.transform(x_trans)
        self.assertTrue(isinstance(res, np.ndarray))

    def test_unknown_function(self):
        with self.assertRaises(ValueError):
            GrakelTransformer(transformation="Dummy")

    def test_unfit_transformer(self):
        g_adapter = GrakelAdapter()
        x_trans = g_adapter.transform(self.Xrandom4d)
        gt = GrakelTransformer(transformation="ShortestPath")
        with self.assertRaises(ValueError):
            gt.transform(x_trans)
