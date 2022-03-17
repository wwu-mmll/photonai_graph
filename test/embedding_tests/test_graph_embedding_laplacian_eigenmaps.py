import unittest
import numpy as np
from photonai_graph.GraphEmbedding.graph_embedding_laplacian_eigenmaps import GraphEmbeddingLaplacianEigenmaps
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels


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
        self.assertEqual((20, 20), np.shape(gembed))

    def test_embedding_laplacian_eigenmaps_complex(self):
        g_embedding = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertFalse(np.iscomplex(gembed).any())

    def test_embedding_laplacian_eigenmaps_complex_random(self):
        X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=100)
        y = get_random_labels(l_type="regression", number_of_labels=100)
        g_embedding = GraphEmbeddingLaplacianEigenmaps()
        g_embedding.fit(X, y)
        embedded = g_embedding.transform(X)
        self.assertFalse(np.iscomplex(embedded).any())

    def test_embedding_laplacian_eigenmaps_3d(self):
        g_embedding = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=3)
        g_embedding.fit(self.X, self.y)
        gembed = g_embedding.transform(self.X)
        self.assertEqual((20, 20, 3), np.shape(gembed))
