import unittest
import numpy as np
import networkx as nx
from dgl.data import MiniGCDataset
from photonai_graph.NeuralNets.GCNModel import GCNClassifierModel
from photonai_graph.GraphUtilities import get_random_labels


class GCNClassifierTests(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.8] = 0
        for i in range(mtrx.shape[0]):
            mtrx[i, :, :, 0][np.diag_indices_from(mtrx[i, :, :, 0])] = 1
        self.Xrandom4d = mtrx
        self.Xrandom_missfit = np.random.rand(20, 20, 5, 2)
        self.X_nx = [nx.erdos_renyi_graph(20, p=0.3)] * 20
        self.y = get_random_labels(number_of_labels=20)
        dgl_mini = MiniGCDataset(20, 10, 20)
        self.X_dgl = dgl_mini.graphs
        self.y_mini = dgl_mini.labels

    def test_gcn_classifier_output_shape(self):
        gcn_clf = GCNClassifierModel(nn_epochs=20)
        gcn_clf.fit(self.Xrandom4d, self.y)
        output = gcn_clf.predict(self.Xrandom4d)
        self.assertTrue(np.array_equal(output.shape, self.y.shape))

    def test_gcn_classifier_output_hidden_dim(self):
        gcn_clf = GCNClassifierModel(hidden_dim=128)
        gcn_clf.fit(self.Xrandom4d, self.y)
        output = gcn_clf.predict(self.Xrandom4d)
        self.assertTrue(np.array_equal(output.shape, self.y.shape))

    def test_gat_classifier_dgl(self):
        gat_clf = GCNClassifierModel(nn_epochs=20)
        gat_clf.fit(self.X_dgl, self.y)
        output = gat_clf.predict(self.X_dgl)
        self.assertTrue(np.array_equal(np.array(output.shape), self.y.shape))
