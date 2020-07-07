from networkx.drawing.nx_agraph import write_dot, read_dot
import stellargraph
from scipy import sparse
import networkx as nx
import numpy as np
import os


def save_networkx_to_file(Graphs, path, output_format="dot", IDs=None):
    # Case 1: a list of networkx graphs as input
    if isinstance(Graphs, list):
        # Case 1.1: an ID is specified
        if isinstance(IDs, list):
            # check they have equal length
            if len(Graphs) == len(IDs):
                # run graph_writing with IDs
                if output_format == "dot":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        write_dot(graph, graph_path)
                elif output_format == "AdjacencyList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_adjlist(graph, graph_path)
                elif output_format == "MultilineAdjacencyList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_multiline_adjlist(graph, graph_path)
                elif output_format == "EdgeList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_edgelist(graph, graph_path)
                elif output_format == "WeightedEdgeList":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_edgelist(graph, graph_path)
                elif output_format == "GEXF":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gexf(graph, graph_path)
                elif output_format == "pickle":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gpickle(graph, graph_path)
                elif output_format == "GML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_gml(graph, graph_path)
                elif output_format == "GraphML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml(graph, graph_path)
                elif output_format == "GraphML-XML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml_xml(graph, graph_path)
                elif output_format == "GraphML-LXML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graphml_lxml(graph, graph_path)
                elif output_format == "YAML":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_yaml(graph, graph_path)
                elif output_format == "graph6":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_graph6(graph, graph_path)
                elif output_format == "PAJEK":
                    for graph, i in zip(Graphs, IDs):
                        graph_filename = "graph_" + str(i)
                        graph_path = os.path.join(path, graph_filename)
                        nx.write_pajek(graph, graph_path)
                else:
                    raise Exception("The output format is not supported. Please check your output format.")
            else:
                # give back error cause unequal length
                raise Exception(
                    'The graph ID list and the list of Graphs are not of equal length. '
                    'Please ensure that they have equal length.')
        # Case 1.2: no ID is specified
        else:
            # run the counter based method if no ID list is specified
            counter = 1
            if output_format == "dot":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    write_dot(graph, graph_path)
                    counter += 1
            elif output_format == "AdjacencyList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_adjlist(graph, graph_path)
                    counter += 1
            elif output_format == "MultilineAdjacencyList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_multiline_adjlist(graph, graph_path)
                    counter += 1
            elif output_format == "EdgeList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_edgelist(graph, graph_path)
                    counter += 1
            elif output_format == "WeightedEdgeList":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_edgelist(graph, graph_path)
                    counter += 1
            elif output_format == "GEXF":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gexf(graph, graph_path)
                    counter += 1
            elif output_format == "pickle":
                for graph, i in zip(Graphs, IDs):
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gpickle(graph, graph_path)
                    counter += 1
            elif output_format == "GML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_gml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML-XML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml_xml(graph, graph_path)
                    counter += 1
            elif output_format == "GraphML-LXML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graphml_lxml(graph, graph_path)
                    counter += 1
            elif output_format == "YAML":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_yaml(graph, graph_path)
                    counter += 1
            elif output_format == "graph6":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_graph6(graph, graph_path)
                    counter += 1
            elif output_format == "PAJEK":
                for graph in Graphs:
                    graph_filename = "graph_" + str(counter)
                    graph_path = os.path.join(path, graph_filename)
                    nx.write_pajek(graph, graph_path)
                    counter += 1
    # Case 2: the input is just a single graph
    if isinstance(Graphs, nx.classes.graph.Graph):
        if output_format == "dot":
            write_dot(Graphs, path)
        elif output_format == "AdjacencyList":
            nx.write_adjlist(Graphs, path)
        elif output_format == "MultilineAdjacencyList":
            nx.write_multiline_adjlist(Graphs, path)
        elif output_format == "EdgeList":
            nx.write_edgelist(Graphs, path)
        elif output_format == "WeightedEdgeList":
            nx.write_edgelist(Graphs, path)
        elif output_format == "GEXF":
            nx.write_gexf(Graphs, path)
        elif output_format == "pickle":
            nx.write_gpickle(Graphs, path)
        elif output_format == "GML":
            nx.write_gml(Graphs, path)
        elif output_format == "GraphML":
            nx.write_graphml(Graphs, path)
        elif output_format == "GraphML-XML":
            nx.write_graphml_xml(Graphs, path)
        elif output_format == "GraphML-LXML":
            nx.write_graphml_lxml(Graphs, path)
        elif output_format == "YAML":
            nx.write_yaml(Graphs, path)
        elif output_format == "graph6":
            nx.write_graph6(Graphs, path)
        elif output_format == "PAJEK":
            nx.write_pajek(Graphs, path)
        else:
            raise Exception("Output format not implemented or recognized. Please check your desired output format.")


def load_file_to_networkx(path, input_format="dot"):
    if isinstance(path, list):
        graph_list = []

        if input_format == "dot":
            for graph in path:
                g = read_dot(graph)
                graph_list.append(g)
        elif input_format == "AdjacencyList":
            for graph in path:
                g = nx.read_adjlist(graph)
                graph_list.append(g)
        elif input_format == "MultilineAdjacencyList":
            for graph in path:
                g = nx.read_multiline_adjlist(graph)
                graph_list.append(g)
        elif input_format == "EdgeList":
            for graph in path:
                g = nx.read_edgelist(graph)
                graph_list.append(g)
        elif input_format == "WeightedEdgeList":
            for graph in path:
                g = nx.read_edgelist(graph)
                graph_list.append(g)
        elif input_format == "GEXF":
            for graph in path:
                g = nx.read_gexf(graph)
                graph_list.append(g)
        elif input_format == "pickle":
            for graph in path:
                g = nx.read_gpickle(graph)
                graph_list.append(g)
        elif input_format == "GML":
            for graph in path:
                g = nx.read_gml(graph)
                graph_list.append(g)
        elif input_format == "GraphML":
            for graph in path:
                g = nx.read_graphml(graph)
                graph_list.append(g)
        elif input_format == "YAML":
            for graph in path:
                g = nx.read_yaml(graph)
                graph_list.append(g)
        elif input_format == "graph6":
            for graph in path:
                g = nx.read_graph6(graph)
                graph_list.append(g)
        elif input_format == "PAJEK":
            for graph in path:
                g = nx.read_pajek(graph)
                graph_list.append(g)

    if isinstance(path, str):
        if input_format == "dot":
            graph_list = read_dot(path)
        elif input_format == "AdjacencyList":
            graph_list = nx.read_adjlist(path)
        elif input_format == "MultilineAdjacencyList":
            graph_list = nx.read_multiline_adjlist(path)
        elif input_format == "EdgeList":
            graph_list = nx.read_edgelist(path)
        elif input_format == "WeightedEdgeList":
            graph_list = nx.read_edgelist(path)
        elif input_format == "GEXF":
            graph_list = nx.read_gexf(path)
        elif input_format == "pickle":
            graph_list = nx.read_gpickle(path)
        elif input_format == "GML":
            graph_list = nx.read_gml(path)
        elif input_format == "GraphML":
            graph_list = nx.read_graphml(path)
        elif input_format == "YAML":
            graph_list = nx.read_yaml(path)
        elif input_format == "graph6":
            graph_list = nx.read_graph6(path)
        elif input_format == "PAJEK":
            graph_list = nx.read_pajek(path)
        else:
            raise Exception('Input format is not supported right now.')

    else:
        raise Exception('Input needs to be a list of paths or a single path.')

    return graph_list


def networkx_to_dense(graphs):
    # convert networkx graphs to dense output
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            np_graph = nx.to_numpy_array(graph)
            graph_list.append(np_graph)
    elif isinstance(graphs, nx.classes.graph.Graph):
        graph_list = nx.to_numpy_array(graphs)
    else:
        raise Exception('Input needs to be either a list of networkx graphs or a networkx graph.')

    return graph_list


def networkx_to_sparse(graphs, format='csr'):
    # convert networkx graphs to sparse output
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            sparse_graph = nx.to_scipy_sparse_matrix(graph, format=format)
            graph_list.append(sparse_graph)
    if isinstance(graphs, nx.classes.graph.Graph):
        graph_list = nx.to_scipy_sparse_matrix(graphs, format=format)
    else:
        raise Exception('Input needs to be a list of networkx graphs or a networkx graph.')

    return graph_list


def networkx_to_stellargraph(graphs, node_features=None):
    # convert networkx graphs to stellargraph graphs
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            sg_graph = stellargraph.StellarGraph.from_networkx(graph, node_features=node_features)
            graph_list.append(sg_graph)
    elif isinstance(graphs, nx.classes.graph.Graph):
        graph_list = stellargraph.StellarGraph.from_networkx(graphs, node_features=node_features)
    else:
        raise Exception('Input needs to be a list of networkx graphs or a networkx graph.')

    return graph_list


def sparse_to_networkx(graphs):
    # convert scipy sparse matrices to networkx graphs
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            nx_graph = nx.from_scipy_sparse_matrix(graph)
            graph_list.append(nx_graph)
    elif isinstance(graphs, sparse.bsr_matrix)\
            or isinstance(graphs, sparse.coo_matrix)\
            or isinstance(graphs, sparse.csc_matrix)\
            or isinstance(graphs, sparse.csr_matrix)\
            or isinstance(graphs, sparse.dia_matrix)\
            or isinstance(graphs, sparse.dok_matrix)\
            or isinstance(graphs, sparse.lil_matrix)\
            or isinstance(graphs, sparse.spmatrix):
        graph_list = nx.from_scipy_sparse_matrix(graphs)
    else:
        raise Exception('Input needs to be a list of sparse matrices or a single sparse matrix.')

    return graph_list


def stellargraph_to_networkx(graphs):
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            nx_graph = graph.to_networkx()
            graph_list.append(nx_graph)
    elif isinstance(graphs, stellargraph.StellarGraph) or isinstance(graphs, stellargraph.StellarDiGraph):
        graph_list = graphs.to_networkx()
    else:
        raise Exception('Input needs to be a list of stellargraph objects or a single stellargraph object.')

    return graph_list

def dense_to_networkx(X, adjacency_axis=0, feature_axis=1, feature_construction="collapse"):

    # convert if Dense is a list of ndarrays
    if isinstance(X, list):
        graph_list = []

        for i in X:
            networkx_graph = nx.from_numpy_matrix(A = i[:, :, adjacency_axis])
            if feature_construction == "collapse":
                features = np.sum(i[:, :, feature_axis], axis=1)
                features = features.reshape((features.shape[0], -1))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features)
            elif feature_construction == "collapse2":
                sum_features = np.sum(i[:, :, feature_axis], axis=1)
                var_features = np.var(i[:, :, feature_axis], axis=1)
                features = np.concatenate((sum_features, var_features))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features)
            graph_list.append(networkx_graph)

        X_converted = graph_list

    # convert if Dense is just a single ndarray
    if isinstance(X, nx.classes.graph.Graph):
        X_converted = nx.from_numpy_matrix(A=X[:, :, adjacency_axis])

    # convert if Dense is an ndarray consisting of multiple arrays
    if isinstance(X, np.ndarray):
        graph_list = []

        for i in range(X.shape[0]):
            networkx_graph = nx.from_numpy_matrix(A=X[i, :, :, adjacency_axis])
            if feature_construction == "collapse":
                features = np.sum(X[i, :, :, feature_axis], axis=1)
                features = features.reshape((features.shape[0], -1))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features, name="collapsed_weight")
            elif feature_construction == "collapse2":
                sum_features = np.sum(X[i, :, :, feature_axis], axis=1)
                var_features = np.var(X[i, :, :, feature_axis], axis=1)
                features = np.column_stack((sum_features, var_features))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features, name="collapsed_weight")
            graph_list.append(networkx_graph)

        X_converted = graph_list


    return X_converted

def dense_to_sparse(graphs, type="coo_matrix", adjacency_axis=None, feature_axis=None):
    if isinstance(graphs, list):
        graph_list = []
        feature_list = []
        if type == "bsr_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.bsr_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.bsr_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.bsr_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "coo_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.coo_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.coo_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.coo_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "csc_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.csc_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.csc_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.csc_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "csr_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.csr_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.csr_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.csr_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "dia_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.dia_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.dia_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.dia_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "dok_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.dok_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.dok_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.dok_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "lil_matrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.lil_matrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.lil_matrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.lil_matrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        elif type == "spmatrix":
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse.spmatrix(graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse.spmatrix(graph[:, :, adjacency_axis])
                    sparse_features = sparse.spmatrix(graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        else:
            raise KeyError('Your desired output format is not supported.'
                           'Please check your output format.')

    elif isinstance(graphs, np.matrix) or isinstance(graphs, np.ndarray):
        graph_list = []
        feature_list = []
        if adjacency_axis is not None and feature_axis is not None:
            # if 4 dimensions you have subjects x values x values x adjacency/features
            if type == "bsr_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.bsr_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.bsr_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                # if 3 dimensions you have values x values x adjacency/features
                # this is because you have given a feature AND adjacency axis
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.bsr_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.bsr_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "coo_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.coo_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.coo_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.coo_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.coo_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "csc_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.csc_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.csc_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.csc_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.csc_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "csr_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.csr_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.csr_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.csr_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.csr_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "dia_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.dia_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.dia_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.dia_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.dia_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "dok_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.dok_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.dok_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.dok_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.dok_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "lil_matrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.lil_matrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.lil_matrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.lil_matrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.lil_matrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')
            if type == "spmatrix":
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.spmatrix(graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse.spmatrix(graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse.spmatrix(graphs[:, :, adjacency_axis])
                    feature_list = sparse.spmatrix(graphs[:, :, feature_axis])
                else:
                    raise Exception('Matrix needs to have 4 or 3 dimensions when axis arguments are given.')

        else:
            if type == "bsr_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.bsr_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.bsr_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            elif type == "coo_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.coo_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.coo_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')

            elif type == "csc_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.csc_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.csc_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            elif type == "csr_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.csr_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.csr_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            elif type == "dia_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.dia_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.dia_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            elif type == "dok_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.dok_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.dok_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            if type == "lil_matrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.lil_matrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.lil_matrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')
            if type == "spmatrix":
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse.spmatrix(graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse.spmatrix(graphs)
                else:
                    raise Exception('Matrix needs to have 3 or 2 dimension when no axis arguments are given.')

    else:
        raise Exception("Input needs to be a numpy array or a list of arrays.")

    return graph_list, feature_list


def sparse_to_dense(graphs, features=None):
    if features is not None:
        if isinstance(graphs, list):
            matrices = []
            for graph, feature in zip(graphs, features):
                graph_mtrx = graph.toarray()
                graph_mtrx = np.reshape((graph_mtrx.shape[0], graph_mtrx.shape[1], -1))
                feature_mtrx = feature.toarray()
                feature_mtrx = np.reshape((feature_mtrx.shape[0], feature_mtrx.shape[1], -1))
                com_mtrx = np.concatenate((graph_mtrx, feature_mtrx), axis=2)
                matrices.append(com_mtrx)
        else:
            try:
                graph_mtrx = graphs.to_array()
                graph_mtrx = np.reshape((graph_mtrx.shape[0], graph_mtrx.shape[1], -1))
                feature_mtrx = features.toarray()
                feature_mtrx = np.reshape((feature_mtrx.shape[0], feature_mtrx.shape[1], -1))
                matrices = np.concatenate((graph_mtrx, feature_mtrx), axis=2)
            except:
                raise Exception('Could not convert matrices.'
                                'Your matrices need to a list or a single sparse matrix.')
    # in this case there is only an adjacency matrix
    else:
        if isinstance(graphs, list):
            matrices = []
            for graph in graphs:
                graph_mtrx = graph.toarray()
                matrices.append(graph_mtrx)
        else:
            try:
                matrices = graphs.toarray()
            except:
                raise Exception('Could not convert matrices.'
                                'Your matrices need to a list or a single sparse matrix.')

    np.asarray(matrices)

    return matrices


def dense_to_stellargraph(graphs):
    # first port to networkx, then to stellargraph
    nx_graphs = dense_to_networkx(graphs)
    sg_graphs = networkx_to_stellargraph(graphs)

    return sg_graphs


def stellargraph_to_dense(graphs):
    # first port to networkx, then to dense
    nx_graphs = stellargraph_to_networkx(graphs)
    matrices = networkx_to_dense(nx_graphs)

    return matrices


def sparse_to_stellargraph(graphs):
    # first port to networkx, then to stellargraph
    nx_graphs = sparse_to_networkx(graphs)
    sg_graphs = networkx_to_stellargraph(nx_graphs)

    return sg_graphs


def stellargraph_to_sparse(graphs, format="csr"):
    # first port to networkx, then to sparse
    nx_graphs = stellargraph_to_networkx(graphs)
    sparse_matrices = networkx_to_sparse(nx_graphs, format=format)

    return sparse_matrices
