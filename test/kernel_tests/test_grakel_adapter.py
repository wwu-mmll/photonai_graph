import unittest
import numpy as np
import networkx as nx
import grakel
from photonai_graph.GraphKernels.grakel_adapter import GrakelAdapter


class GrakelAdapterTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.Xrandom4d = mtrx
        self.Xrandom_missfit = np.random.rand(20, 20, 5, 2)
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        self.y = np.ones(20)

    def test_grakel_adapter_type_list(self):
        g_adapter = GrakelAdapter()
        g_adapter.fit(self.Xrandom4d, self.y)
        x_trans = g_adapter.transform(self.Xrandom4d)
        self.assertEqual(type(x_trans), list)

    def test_unexpected_input(self):
        g_adapter = GrakelAdapter(feature_axis=0)
        g_in = np.array([[1, 0, 0], [-1, 3, 4]])
        with self.assertRaises(ValueError):
            g_adapter.transform(g_in[np.newaxis, ..., np.newaxis])

    def test_unexpected_input_type(self):
        with self.assertRaises(ValueError):
            GrakelAdapter(input_type="non-sens")

    def test_grakel_adapter_type_graph(self):
        g_adapter = GrakelAdapter()
        x_trans = g_adapter.transform(self.Xrandom4d)
        self.assertEqual(type(x_trans[0]), grakel.Graph)

    def test_grakel_adapter_integration_kernel_shape(self):
        g_adapter = GrakelAdapter()
        g_kernel = grakel.kernels.PyramidMatch()
        x_trans = g_adapter.transform(self.Xrandom4d)
        x_kern = g_kernel.fit_transform(x_trans)
        self.assertEqual(x_kern.shape, (20, 20))

    def test_grakel_adapter_adjacency_shape(self):
        with self.assertRaises(Exception):
            g_adapter = GrakelAdapter()
            g_adapter.transform(self.Xrandom_missfit)

    def test_grakel_adapter_networkx(self):
        g_adapter = GrakelAdapter(input_type="networkx")
        g_kernel = grakel.kernels.RandomWalk()
        x_trans = g_adapter.transform(self.X_nx)
        x_kern = g_kernel.fit_transform(x_trans)
        self.assertEqual(x_kern.shape, (20, 20))
