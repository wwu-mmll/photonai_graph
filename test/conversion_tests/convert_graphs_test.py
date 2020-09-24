import unittest
import dgl
import networkx as nx
import warnings
from photonai_graph.GraphConversions import convert_graphs


class ConvertGraphsTest(unittest.TestCase):

    def setUp(self):
        # create networkx graphs
        nx_graph = nx.cycle_graph(5)
        self.nx_graph_list = [nx_graph] * 10

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            convert_graphs(self.nx_graph_list, input_format="networkx",
                           output_format="networkx")
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue("desired format" in str(w[-1].message))

    def test_nonsense_conversion(self):
        with self.assertRaises(TypeError):
            convert_graphs(self.nx_graph_list, input_format="edge_list",
                           output_format="networkx")

    def test_networkx_to_dgl(self):
        g = convert_graphs(self.nx_graph_list, input_format="networkx",
                           output_format="dgl")
        self.assertEqual(type(g[0]), dgl.DGLGraph)


if __name__ == '__main__':
    unittest.main()
