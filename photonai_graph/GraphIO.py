import os

from networkx.drawing.nx_pydot import write_dot, read_dot
import networkx as nx
import numpy as np
import yaml

output_formats = {
    "dot": write_dot,
    "AdjacencyList": nx.write_adjlist,
    "MultilineAdjacencyList": nx.write_multiline_adjlist,
    "EdgeList": nx.write_edgelist,
    "WeightedEdgeList": nx.write_weighted_edgelist,
    "GEXF": nx.write_gexf,
    "pickle": nx.write_gpickle,
    "GLM": nx.write_gml,
    "GraphML": nx.write_graphml,
    "GraphML-XML": nx.write_graphml_xml,
    "GraphML-LXML": nx.write_graphml_lxml,
    "YAML": yaml.dump,
    "graph6": nx.write_graph6,
    "PAJEK": nx.write_pajek
}

input_formats = {
    "dot": read_dot,
    "AdjacencyList": nx.read_adjlist,
    "MultilineAdjacencyList": nx.read_multiline_adjlist,
    "EdgeList": nx.read_edgelist,
    "WeightedEdgeList": nx.read_edgelist,
    "GEXF": nx.read_gexf,
    "pickle": nx.read_gpickle,
    "GML": nx.read_gml,
    "GraphML": nx.read_graphml,
    "YAML": yaml.load,
    "graph6": nx.read_graph6,
    "PAJEK": nx.read_pajek
}


def save_networkx_to_file(graphs, path, output_format="dot", ids=None):
    """Saves networkx graph(s) to a file

        Parameters
        ----------
        graphs: a list of networkx graphs or single networkx graph
            list of graphs you want to save
        path: str
            path where to save the data as a string
        output_format: str
            output format in which the graphs should be saved. See output_formats
            at top of the file for more information on valid output formats
        ids: list, default=None
            a list of ids after which to name each graph in the file. If no list is
            specified the graphs are just numbered as graph_x

    """
    def save_single_networkx_to_file(graph, path, output_format="dot"):
        """internal helper function

        Parameters
        ----------
        graph
            networkx graph object
        path
            path to save the graph to
        output_format
            desired output format
        """
        if not isinstance(graph, nx.classes.graph.Graph):
            raise ValueError(f"Got unknown object for serialization: {type(graph)}")
        output_formats[output_format](graph, path)

    if output_format not in output_formats:
        raise ValueError("The output format is not supported. Please check your output format.")

    if not isinstance(graphs, list):
        return save_single_networkx_to_file(graph=graphs, path=path, output_format=output_format)

    # Check if we have got a list of ids
    if ids is None or not isinstance(ids, list):
        ids = np.arange(len(graphs))

    # check if id and graphs they have equal length
    if len(graphs) != len(ids):
        raise ValueError(
            'The photonai_graph ID list and the list of Graphs are not of equal length. '
            'Please ensure that they have equal length.')

    for graph, i in zip(graphs, ids):
        graph_filename = "graph_" + str(i)
        graph_path = os.path.join(path, graph_filename)
        save_single_networkx_to_file(graph=graph, path=graph_path, output_format=output_format)


def load_file_to_networkx(path, input_format="dot"):
    """load graph into networkx from a file

        Parameters
        ----------
        path: list, str
            path(s) where graphs are stored
        input_format:
            format in which these graphs are stored

        Returns
        -------
        list
            List of loaded networkx graphs
    """
    if isinstance(path, str):
        path = [path]

    graph_list = []
    if input_format not in input_formats:
        raise KeyError("Input format is not supported right now.")

    for graph in path:
        g = input_formats[input_format](graph)
        graph_list.append(g)

    return graph_list
