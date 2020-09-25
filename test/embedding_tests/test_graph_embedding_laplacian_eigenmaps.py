import unittest
import numpy as np
from photonai_graph.GraphEmbedding.graph_embedding_laplacian_eigenmaps import GraphEmbeddingLaplacianEigenmaps


class LaplacianEigenmapsTest(unittest.TestCase):

    def setUp(self):
        mtrx = np.random.rand(20, 20, 20, 2)
        mtrx[:, :, :, 0][mtrx[:, :, :, 0] <= 0.9] = 0
        self.X = mtrx
        self.y = np.ones(20)

    def test_embedding_laplacian_eigenmaps(self):
        g_embedding = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual(np.shape(gembed), (20, 20))

    def test_embedding_laplacian_eigenmaps_complex(self):
        g_embedding = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        bool_mask = [False] * 20
        bool_array = np.iscomplex(gembed[0])
        comp = bool_mask == bool_array
        comp = comp.all()
        self.assertTrue(comp)

    def test_embedding_laplacian_eigenmaps_3d(self):
        g_embedding = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=3)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual(np.shape(gembed), (20, 20, 3))
