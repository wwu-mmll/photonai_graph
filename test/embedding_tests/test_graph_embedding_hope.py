import unittest
import numpy as np
from photonai_graph.GraphEmbedding.graph_embedding_hope import GraphEmbeddingHOPE


class HOPETest(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.X = mtrx
        self.y = np.ones(20)

    def test_embedding_hope(self):
        g_embedding = GraphEmbeddingHOPE()
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual((20, 40), np.shape(gembed))
        orig_transformed = g_embedding.orig_transformed
        for graph in range(self.X.shape[0]):
            self.assertTrue(np.array_equal(np.squeeze(np.reshape(orig_transformed[graph, ...], (-1, 1))),
                                           gembed[graph, ...]))

    def test_embedding_hope_2d(self):
        g_embedding = GraphEmbeddingHOPE(embedding_dimension=2)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual((20, 20, 2), np.shape(gembed))

    def test_embedding_hope_4d(self):
        g_embedding = GraphEmbeddingHOPE(embedding_dimension=4)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual((20, 20, 4), np.shape(gembed))
