import unittest
from scipy import sparse
import networkx as nx
from photonai_graph.GraphConversions import sparse_to_networkx


class SparseToNetworkxTest(unittest.TestCase):

    def setUp(self):
        check_matrix = sparse.csr_matrix([[0, 1, 0, 0, 1],
                                          [1, 0, 1, 0, 0],
                                          [0, 1, 0, 1, 0],
                                          [0, 0, 1, 0, 1],
                                          [1, 0, 0, 1, 0]])
        self.single_matrix = check_matrix
        self.matrix_list = [check_matrix] * 10
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_list(self):
        lst = sparse_to_networkx(self.matrix_list)
        self.assertEqual(type(lst), list)

    def test_list_element(self):
        lst = sparse_to_networkx(self.matrix_list)
        self.assertEqual(type(lst[0]), nx.classes.Graph)

    def test_single(self):
        single = sparse_to_networkx(self.single_matrix)
        self.assertEqual(type(single), nx.classes.Graph)

    def test_nodes(self):
        single = sparse_to_networkx(self.single_matrix)
        self.assertEqual(single.number_of_nodes(), 5)

    def test_edges(self):
        single = sparse_to_networkx(self.single_matrix)
        self.assertEqual(single.number_of_edges(), 5)

    def test_adjacency(self):
        single = sparse_to_networkx(self.single_matrix)
        smat = nx.adjacency_matrix(single)
        self.assertTrue((smat != self.single_matrix).nnz == 0)

    def test_nonsense(self):
        with self.assertRaises(Exception):
            sparse_to_networkx(self.edge_dict)
