import unittest
import numpy as np
import networkx as nx
from dgl.data import MiniGCDataset
from photonai_graph.NeuralNets.NNClassifier.gat_classifier import GATClassifierModel
from photonai_graph.GraphUtilities import get_random_labels


class GATClassifierTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.Xrandom4d = mtrx
        self.Xrandom_missfit = np.random.rand(20, 20, 5, 2)
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        dgl_mini = MiniGCDataset(20, 10, 20)
        self.X_dgl = dgl_mini.graphs
        self.y = get_random_labels(number_of_labels=20)
        self.y_mini = dgl_mini.labels

    def test_gat_classifier_output_shape(self):
        gat_clf = GATClassifierModel(nn_epochs=20)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))

    def test_gat_classifier_output_hidden_dim(self):
        gat_clf = GATClassifierModel(hidden_dim=128)
        gat_clf.fit(self.Xrandom4d, self.y)
        output = gat_clf.predict(self.Xrandom4d)
        self.assertEqual(output.shape, (20, 1))

    def test_gat_classifier_nx_graphs(self):
        gat_clf = GATClassifierModel(nn_epochs=20)
        gat_clf.fit(self.X_nx, self.y)
        output = gat_clf.predict(self.X_nx)
        self.assertEqual(output.shape, (20, 1))

    def test_gat_classifier_dgl(self):
        gat_clf = GATClassifierModel(nn_epochs=20)
        gat_clf.fit(self.X_dgl, self.y)
        output = gat_clf.predict(self.X_dgl)
        self.assertEqual(output.shape, (20, 1))
