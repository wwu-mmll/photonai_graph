import unittest
import dgl
import networkx as nx
from photonai_graph.GraphConversions import dgl_to_networkx


class DglToNetworkxTest(unittest.TestCase):

    def setUp(self):
        # create dgl graph and nonsense input
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(3)
        dgl_graph.add_edges([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])
        self.graphs = [dgl_graph] * 10
        self.edge_dict = {(1, 0): 1, (2, 0): 1}

    def test_nonsense_input(self):
        with self.assertRaises(Exception):
            dgl_to_networkx(self.edge_dict)

    def test_list_type(self):
        g = dgl_to_networkx(self.graphs, node_attrs=[], edge_attrs=[])
        self.assertEqual(type(g), list)

    def test_output_format_multigraph(self):
        g = dgl_to_networkx(self.graphs, node_attrs=[], edge_attrs=[])
        self.assertEqual(type(g[0]), nx.classes.MultiDiGraph)

    def test_output_graph_num_nodes(self):
        g = dgl_to_networkx(self.graphs, node_attrs=[], edge_attrs=[])
        graph = g[0]
        self.assertEqual(graph.number_of_nodes(), 3)

    def test_output_graph_num_edges(self):
        g = dgl_to_networkx(self.graphs, node_attrs=[], edge_attrs=[])
        graph = g[0]
        self.assertEqual(graph.number_of_edges(), 6)


if __name__ == '__main__':
    unittest.main()
