import unittest
import numpy as np
import networkx as nx
from photonai_graph.NeuralNets.GCNModel import GCNRegressorModel
from photonai_graph.GraphUtilities import get_random_labels


class GCNRegressorTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        for i in range(mtrx.shape[0]):
            mtrx[i, :, :, 0][np.diag_indices_from(mtrx[i, :, :, 0])] = 1
        self.Xrandom4d = mtrx
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        self.y = get_random_labels(number_of_labels=20)

    def test_gcn_regressor_output_shape(self):
        gat_clf = GCNRegressorModel(nn_epochs=20)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, self.y.shape)

    def test_gcn_regressor_output_hidden_dim(self):
        gat_clf = GCNRegressorModel(hidden_dim=128)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, self.y.shape)
