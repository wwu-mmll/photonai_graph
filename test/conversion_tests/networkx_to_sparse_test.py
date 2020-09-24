import unittest
import networkx as nx
import scipy.sparse
from photonai_graph.GraphConversions import networkx_to_sparse


class NetworkxToSparseTest(unittest.TestCase):

    def setUp(self):
        self.graph = nx.cycle_graph(20)
        self.graphs = [nx.cycle_graph(20)] * 10
        self.check_graph = nx.cycle_graph(5)
        self.check_matrix = scipy.sparse.csr_matrix([[0, 1, 0, 0, 1],
                                                     [1, 0, 1, 0, 0],
                                                     [0, 1, 0, 1, 0],
                                                     [0, 0, 1, 0, 1],
                                                     [1, 0, 0, 1, 0]])

    def test_list(self):
        mtrx = networkx_to_sparse(self.graphs)
        self.assertEqual(type(mtrx), list)

    def test_np_type(self):
        mtrx = networkx_to_sparse(self.graphs)
        self.assertEqual(type(mtrx[0]), scipy.sparse.csr_matrix)

    def test_single_graph(self):
        mtrx = networkx_to_sparse(self.graph)
        self.assertEqual(type(mtrx), scipy.sparse.csr_matrix)

    def test_matrix_shape(self):
        mtrx = networkx_to_sparse(self.graph)
        self.assertEqual(mtrx.shape, (20, 20))

    def test_check_graph(self):
        mtrx = networkx_to_sparse(self.check_graph)
        self.assertTrue((mtrx!=self.check_matrix).nnz==0)

    def test_check_nonsense_input(self):
        with self.assertRaises(ValueError):
            mtrx = networkx_to_sparse(self.check_matrix)


if __name__ == '__main__':
    unittest.main()
