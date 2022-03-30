import unittest
import dgl
import networkx as nx
from scipy import sparse
import numpy as np
from photonai_graph.GraphConversions import check_dgl


class DglToNetworkxTest(unittest.TestCase):

    def setUp(self):
        # create dgl graphs
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(3)
        dgl_graph.add_edges([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])
        self.dgl_graph_list = [dgl_graph] * 10
        # create networkx graphs
        nx_graph = nx.cycle_graph(5)
        self.nx_graph_list = [nx_graph] * 10
        # create scipy matrix
        sparse_matrix = sparse.csr_matrix([[0, 1, 1],
                                           [1, 0, 1],
                                           [1, 1, 0]])
        self.sp_matrix_list = [sparse_matrix] * 10
        # create numpy matrix
        array = np.array([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0]])
        m4d_array = array[np.newaxis, :, :, np.newaxis]
        individuals_array = np.repeat(m4d_array, 5, axis=0)
        modality_array = np.repeat(individuals_array, 2, axis=3)
        self.np_4d_array = modality_array
        self.np_5d_array = modality_array[:, :, :, :, np.newaxis]
        # create nonsense input
        self.edge_dict = {(1, 0): 1, (2, 0): 1}
        self.np_list = [np.ones((3, 3))] * 10

    def test_nonsense_input(self):
        with self.assertRaises(TypeError):
            check_dgl(self.edge_dict)

    def test_list_dgl(self):
        g = check_dgl(self.dgl_graph_list)
        self.assertEqual(len(g), 10)

    def test_dgl_output_num_nodes(self):
        g = check_dgl(self.dgl_graph_list)
        self.assertEqual(g[0].number_of_nodes(), 3)

    def test_dgl_output_num_edges(self):
        g = check_dgl(self.dgl_graph_list)
        self.assertEqual(g[0].number_of_edges(), 6)

    def test_numpy_list_exception(self):
        with self.assertRaises(Exception):
            check_dgl(self.np_list)

    def test_list_np4d(self):
        g = check_dgl(self.np_4d_array, adjacency_axis=0, feature_axis=1)
        self.assertEqual(len(g), 5)

    def test_np4d_conversion(self):
        g = check_dgl(self.np_4d_array, adjacency_axis=0, feature_axis=1)
        self.assertEqual(type(g[0]), dgl.DGLGraph)

    def test_np4d_output_num_nodes(self):
        g = check_dgl(self.np_4d_array, adjacency_axis=0, feature_axis=1)
        self.assertEqual(g[0].number_of_nodes(), 3)

    def test_np4d_output_num_edges(self):
        g = check_dgl(self.np_4d_array, adjacency_axis=0, feature_axis=1)
        self.assertEqual(g[0].number_of_edges(), 6)

    def test_np5d_error(self):
        with self.assertRaises(ValueError):
            check_dgl(self.np_5d_array, adjacency_axis=0, feature_axis=1)
