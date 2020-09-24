import unittest
import numpy as np
import networkx as nx
from photonai_graph.NeuralNets.NNRegressor.sgc_regressor import SGConvRegressorModel
from photonai_graph.GraphUtilities import get_random_labels


class SGConvRegressorTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.Xrandom4d = mtrx
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        self.y = get_random_labels(number_of_labels=20)

    def test_gat_regressor_output_shape(self):
        gat_clf = SGConvRegressorModel(nn_epochs=20)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))

    def test_gat_regressor_output_hidden_dim(self):
        gat_clf = SGConvRegressorModel(hidden_dim=128)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))