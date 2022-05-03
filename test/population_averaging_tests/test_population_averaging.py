import unittest

import numpy as np
import networkx as nx
from photonai_graph.PopulationAveragingTransform import PopulationAveragingTransform


class PopulationAveragingTests(unittest.TestCase):

    def setUp(self):
        # Randomly create 10 barabasi albert graphs and store the adjacency as numpy array
        self.graphs = np.array([nx.adjacency_matrix(nx.barabasi_albert_graph(20, 2)).todense() for _ in range(10)])
        # Generate random graphs for 10 subjects containing 3 measurements
        self.other_graphs = np.random.rand(10, 20, 20, 3)

    def test_averaging(self):
        mean = np.squeeze(np.mean(self.graphs, axis=0))
        pat = PopulationAveragingTransform()
        pat.fit(self.graphs[..., np.newaxis], 0)
        self.assertTrue(np.array_equal(mean, pat.learned_mean))

        transformed = pat.transform(self.other_graphs)
        for subject in range(transformed.shape[0]):
            self.assertTrue(np.array_equal(mean, transformed[subject, ..., 0]))
            self.assertTrue(np.array_equal(self.other_graphs[subject, ..., 2], transformed[subject, ..., 1]))

    def test_unfitted_transformer(self):
        pat = PopulationAveragingTransform()
        with self.assertRaises(ValueError):
            pat.transform(self.other_graphs)

    def test_wrong_input_dimensionality(self):
        dummy_input = np.ones((10, 20, 20))
        pat = PopulationAveragingTransform()
        with self.assertRaises(ValueError):
            pat.fit(dummy_input, [])
        pat.fit(dummy_input[..., np.newaxis], [])
        with self.assertRaises(ValueError):
            pat.transform(dummy_input)
