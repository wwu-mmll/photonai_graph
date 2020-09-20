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
        g_embedding = GraphEmbeddingHOPE(embedding_dimension=1)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual(np.shape(gembed), (20, 40, 1))

    def test_embedding_hope_2d(self):
        g_embedding = GraphEmbeddingHOPE(embedding_dimension=2)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual(np.shape(gembed), (20, 20, 2, 1))

    def test_embedding_hope_4d(self):
        g_embedding = GraphEmbeddingHOPE(embedding_dimension=4)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual(np.shape(gembed), (20, 20, 4, 1))


if __name__ == '__main__':
    unittest.main()
