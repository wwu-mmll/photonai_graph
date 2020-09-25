import unittest
import numpy as np
import networkx as nx
from photonai_graph.NeuralNets.NNClassifier.gcn_classifier import GCNClassifierModel
from photonai_graph.GraphUtilities import get_random_labels


class GCNClassifierTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.Xrandom4d = mtrx
        self.Xrandom_missfit = np.random.rand(20, 20, 5, 2)
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        self.y = get_random_labels(number_of_labels=20)

    def test_gcn_classifier_output_shape(self):
        gcn_clf = GCNClassifierModel(nn_epochs=20)
        gcn_clf.fit(self.Xrandom4d, self.y)
        output = gcn_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))

    def test_gcn_classifier_output_hidden_dim(self):
        gcn_clf = GCNClassifierModel(hidden_dim=128)
        gcn_clf.fit(self.Xrandom4d, self.y)
        output = gcn_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))
