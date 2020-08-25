from networkx.drawing.nx_agraph import write_dot, read_dot
import stellargraph
from scipy import sparse
import networkx as nx
import numpy as np
import os

output_formats = {
    "dot": write_dot,
    "AdjacencyList": nx.write_adjlist,
    "MultilineAdjacencyList": nx.write_multiline_adjlist,
    "EdgeList": nx.write_edgelist,
    "WeightedEdgeList": nx.write_edgelist,  # todo: is this intended?
    "GEXF":  nx.write_gexf,
    "pickle": nx.write_gpickle,
    "GLM": nx.write_gml,
    "GraphML": nx.write_graphml,
    "GraphML-XML": nx.write_graphml_xml,
    "GraphML-LXML": nx.write_graphml_lxml,
    "YAML": nx.write_yaml,
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
    "YAML": nx.read_yaml,
    "graph6": nx.read_graph6,
    "PAJEK": nx.read_pajek
}

sparse_types = {
    "bsr_matrix": sparse.bsr_matrix,
    "coo_matrix": sparse.coo_matrix,
    "csc_matrix": sparse.csc_matrix,
    "csr_matrix": sparse.csr_matrix,
    "dia_matrix": sparse.dia_matrix,
    "dok_matrix": sparse.dok_matrix,
    "lil_matrix": sparse.lil_matrix,
    "spmatrix": sparse.spmatrix
}


def save_networkx_to_file(Graphs, path, output_format="dot", IDs=None):
    # Case 1: a list of networkx graphs as input
    if isinstance(Graphs, list):
        # Check if we have got a list of ids
        if IDs is None or not isinstance(IDs, list):
            IDs = np.arange(len(Graphs))
        # check if id and graphs they have equal length
        if len(Graphs) == len(IDs):
            # run graph_writing with IDs
            if output_format in output_formats:
                for graph, i in zip(Graphs, IDs):
                    graph_filename = "graph_" + str(i)
                    graph_path = os.path.join(path, graph_filename)
                    output_formats[output_format](graph, graph_path)
            else:
                raise Exception("The output format is not supported. Please check your output format.")
        else:
            # give back error cause unequal length
            raise Exception(
                'The photonai_graph ID list and the list of Graphs are not of equal length. '
                'Please ensure that they have equal length.')

    # Case 2: the input is just a single photonai_graph
    if isinstance(Graphs, nx.classes.graph.Graph):
        if output_format in output_formats:
            output_formats[output_format](Graphs, path)
        else:
            raise Exception("Output format not implemented or recognized. Please check your desired output format.")


def load_file_to_networkx(path, input_format="dot"):
    if isinstance(path, str):
        path = [path]

    graph_list = []
    if input_format in input_formats:
        for graph in path:
            g = input_formats[input_format](graph)
            graph_list.append(g)
    else:
        raise Exception("Input format is not supported right now.")

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
        raise Exception('Input needs to be either a list of networkx graphs or a networkx photonai_graph.')

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
        raise Exception('Input needs to be a list of networkx graphs or a networkx photonai_graph.')

    return graph_list


def networkx_to_dgl(graphs, node_attrs=None, edge_attrs=None):
    # convert networkx graphs to dgl graphs
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            g = dgl.DGLGraph.from_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
            graph_list.append(g)
    elif isinstance(graphs, np.ndarray):
        graph_list = []
        for graph in range(graphs.shape[0]):
            g = dgl.DGLGraph.from_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
            graph_list.append(g)
    elif isinstance(graphs, nx.classes.graph.Graph):
        graph_list = dgl.DGLGraph.from_networkx(graphs)
    else:
        raise Exception('networkx_to_dgl only implemented for list, ndarrays or single networkx graph')

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
        raise Exception('Input needs to be a list of networkx graphs or a networkx photonai_graph.')

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
            # todo: duplicated code
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
            # todo: duplicated code
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
        if type in sparse_types:
            for graph in graphs:
                if np.ndim(graph) == 2:
                    sparse_graph = sparse_types[type](graph)
                    graph_list.append(sparse_graph)
                if np.ndim(graph) == 3:
                    sparse_adjacency = sparse_types[type](graph[:, :, adjacency_axis])
                    sparse_features = sparse_types[type](graph[:, :, feature_axis])
                    graph_list.append(sparse_adjacency)
                    feature_list.append(sparse_features)
        else:
            raise KeyError("Your desired output format is not supported.\nPlease check your output format.")

    elif isinstance(graphs, np.matrix) or isinstance(graphs, np.ndarray):
        graph_list = []
        feature_list = []
        if adjacency_axis is not None and feature_axis is not None:
            if type in sparse_types:
                # if 4 dimensions you have subjects x values x values x adjacency/features
                if np.ndim(graphs) == 4:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse_types[type](graphs[i, :, :, adjacency_axis])
                        graph_list.append(adjacency)
                        feature = sparse_types[type](graphs[i, :, :, feature_axis])
                        feature_list.append(feature)
                elif np.ndim(graphs) == 3:
                    graph_list = sparse_types[type](graphs[:, :, adjacency_axis])
                    feature_list = sparse_types[type](graphs[:, :, feature_axis])
                else:
                    raise Exception("Matrix needs to have 4 or 3 dimensions when axis arguments are given.")

        else:
            if type in sparse_types:
                if np.ndim(graphs) == 3:
                    for i in range(graphs.shape[0]):
                        adjacency = sparse_types[type](graphs[i, :, :])
                        graph_list.append(adjacency)
                if np.ndim(graphs) == 2:
                    graph_list = sparse_types[type](graphs)
                else:
                    raise Exception("Matrix needs to have 3 or 2 dimension when no axis arguments are given.")

    else:
        raise Exception("Input needs to be a numpy array or a list of arrays.")

    return graph_list, feature_list


def dense_to_dgl(graphs, adjacency_axis=None, feature_axis=None):
    # this function converts dense matrices to dgl graphs
    if adjacency_axis is None:
        raise Exception('dense to dgl not implemented without adjacency axis')
    else:
        sparse_mtrx = dense_to_sparse(graphs, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        graph_list = sparse_to_dgl(sparse_mtrx)
    return graph_list


def sparse_to_dense(graphs, features=None):
    if features is not None:
        if isinstance(graphs, list):
            matrices = []
            for graph, feature in zip(graphs, features):
                graph_mtrx = graph.toarray()
                graph_mtrx = np.reshape((graph_mtrx.shape[0], graph_mtrx.shape[1], -1))
                feature_mtrx = feature.toarray()
                # todo: duplicated code
                feature_mtrx = np.reshape((feature_mtrx.shape[0], feature_mtrx.shape[1], -1))
                com_mtrx = np.concatenate((graph_mtrx, feature_mtrx), axis=2)
                matrices.append(com_mtrx)
        else:
            try:
                graph_mtrx = graphs.to_array()
                # todo: duplicated code
                graph_mtrx = np.reshape((graph_mtrx.shape[0], graph_mtrx.shape[1], -1))
                feature_mtrx = features.toarray()
                feature_mtrx = np.reshape((feature_mtrx.shape[0], feature_mtrx.shape[1], -1))
                matrices = np.concatenate((graph_mtrx, feature_mtrx), axis=2)
            except Exception as e:
                # If you simply catch this exception and raise a custom one, the original error message is lost.
                # The user almost always wants to know the real reason of this error for debugging.
                print('Could not convert matrices.'
                      'Your matrices need to a list or a single sparse matrix.')
                raise e
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
            except Exception as e:
                print('Could not convert matrices.'
                      'Your matrices need to a list or a single sparse matrix.')
                raise e

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


def sparse_to_dgl(graphs, adjacency_axis=0):
    # take dense and make them long
    if isinstance(graphs, tuple):
        graph_list = []
        for adj, feat in zip(graphs[0], graphs[1]):
            g = dgl.DGLGraph()
            g.from_scipy_sparse_matrix(spmat=adj)
            graph_list.append(g)
    else:
        raise Exception('Expected tuple as input.')
    return graph_list


def stellargraph_to_sparse(graphs, format="csr"):
    # first port to networkx, then to sparse
    nx_graphs = stellargraph_to_networkx(graphs)
    sparse_matrices = networkx_to_sparse(nx_graphs, format=format)

    return sparse_matrices


def check_dgl(graphs, adjacency_axis=None, feature_axis=None):
    # this functions checks the input and converts it to dgl format
    if isinstance(graphs, list):
        if isinstance(graphs[0], nx.classes.graph.Graph):
            dgl_graphs = networkx_to_dgl(graphs)
        elif isinstance(graphs[0], sparse.spmatrix) \
            or isinstance(graphs[0], sparse.bsr_matrix) \
            or isinstance(graphs[0], sparse.lil_matrix) \
            or isinstance(graphs[0], sparse.csc_matrix) \
            or isinstance(graphs[0], sparse.coo_matrix) \
            or isinstance(graphs[0], sparse.csr_matrix) \
            or isinstance(graphs[0], sparse.dok_matrix) \
            or isinstance(graphs[0], sparse.dia_matrix):
            dgl_graphs = sparse_to_dgl(graphs)
        elif isinstance((graphs[0]), dgl.DGLGraph):
            dgl_graphs = graphs
        else:
            raise Exception('Can only handle lists of networkx graphs or scipy matrices')
    elif isinstance(graphs, np.ndarray) or isinstance(graphs, np.matrix):
        if np.ndim(graphs) == 1:
            if isinstance(graphs[0], nx.classes.graph.Graph):
                dgl_graphs = networkx_to_dgl(graphs)
            elif isinstance(graphs[0], sparse.spmatrix) \
                    or isinstance(graphs[0], sparse.bsr_matrix) \
                    or isinstance(graphs[0], sparse.lil_matrix) \
                    or isinstance(graphs[0], sparse.csc_matrix) \
                    or isinstance(graphs[0], sparse.coo_matrix) \
                    or isinstance(graphs[0], sparse.csr_matrix) \
                    or isinstance(graphs[0], sparse.dok_matrix) \
                    or isinstance(graphs[0], sparse.dia_matrix):
                dgl_graphs = sparse_to_dgl(graphs)
            elif isinstance((graphs[0]), dgl.DGLGraph):
                dgl_graphs = graphs
            else:
                raise Exception('Can only handle ndarrays of networkx graphs or scipy matrices')
        elif np.ndim(graphs) > 1:
            dgl_graphs = dense_to_dgl(graphs, adjacency_axis, feature_axis)
        else:
            raise Exception('numpy matrix must have between one and four dimensions')
    else:
        raise TypeError('can only handle np arrays of lists as input')

    return dgl_graphs
