import unittest
from scipy import sparse
import dgl
from photonai_graph.GraphConversions import dgl_to_sparse


class DglToSparseTest(unittest.TestCase):

    def setUp(self):
        # create dgl graph and nonsense input
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(3)
        dgl_graph.add_edges([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])
        self.graphs = [dgl_graph] * 10
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_nonsense_input(self):
        with self.assertRaises(Exception):
            dgl_to_sparse(self.edge_dict)

    def test_list_type(self):
        g = dgl_to_sparse(self.graphs)
        self.assertEqual(type(g), list)

    def test_output_format_csr(self):
        g = dgl_to_sparse(self.graphs, fmt="csr")
        self.assertEqual(type(g[0]), sparse.csr_matrix)

    def test_output_format_coo(self):
        g = dgl_to_sparse(self.graphs, fmt="coo")
        self.assertEqual(type(g[0]), sparse.coo_matrix)
