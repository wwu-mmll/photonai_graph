import unittest
import numpy as np
import scipy.sparse
from photonai_graph.GraphUtilities import get_random_connectivity_data


class DrawConnectogramTests(unittest.TestCase):

    def setUp(self):
        self.np_control = np.random.rand(20, 20, 20, 2)

    def test_matrix_type(self):
        mtrx = get_random_connectivity_data(out_type="dense", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        self.assertEqual(type(self.np_control), type(mtrx))

    def test_matrix_shape(self):
        mtrx = get_random_connectivity_data(out_type="dense", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        self.assertEqual(np.shape(self.np_control), np.shape(mtrx))

    def test_sparse_type(self):
        mtrx = get_random_connectivity_data(out_type="sparse", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        self.assertEqual(type(mtrx), list)

    def test_sparse_len_individuals(self):
        mtrx = get_random_connectivity_data(out_type="sparse", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        self.assertEqual(len(mtrx), 20)

    def test_sparse_len_modality(self):
        mtrx = get_random_connectivity_data(out_type="sparse", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        self.assertEqual(len(mtrx[0]), 2)

    def test_sparse_type_sparse(self):
        mtrx = get_random_connectivity_data(out_type="sparse", number_of_nodes=20,
                                            number_of_individuals=20, number_of_modalities=2)
        individual_one = mtrx[0]
        self.assertEqual(type(individual_one[0]), scipy.sparse.coo_matrix)

    def test_wrong_type(self):
        with self.assertRaises(NotImplementedError):
            get_random_connectivity_data(out_type="networkx", number_of_nodes=20,
                                         number_of_individuals=20, number_of_modalities=2)
