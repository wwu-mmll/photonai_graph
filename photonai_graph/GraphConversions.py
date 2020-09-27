from networkx.drawing.nx_pydot import write_dot, read_dot
from scipy import sparse
import networkx as nx
import numpy as np
import dgl
import os
import warnings

output_formats = {
    "dot": write_dot,
    "AdjacencyList": nx.write_adjlist,
    "MultilineAdjacencyList": nx.write_multiline_adjlist,
    "EdgeList": nx.write_edgelist,
    "WeightedEdgeList": nx.write_weighted_edgelist,
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

    # run graph_writing with IDs
    if output_format not in output_formats:
        raise ValueError("The output format is not supported. Please check your output format.")

    # Case 1: a list of networkx graphs as input
    if isinstance(graphs, list):

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
            output_formats[output_format](graph, graph_path)

    # Case 2: the input is just a single photonai_graph
    elif isinstance(graphs, nx.classes.graph.Graph):
        output_formats[output_format](graphs, path)

    else:
        raise ValueError('Your input is not supported right now.\nPlease provide a list of graphs or a single graph')


def load_file_to_networkx(path, input_format="dot"):
    """load graph into networkx from a file

        Parameters
        ----------
        path: list, str
            path(s) where graphs are stored
        input_format:
            format in which these graphs are stored
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


def networkx_to_dense(graphs):
    """convert a networkx graph to a numpy array

        Parameters
        ----------
            graphs: list of graphs

        :returns
            list of numpy arrays of a single numpy array
    """
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            np_graph = nx.to_numpy_array(graph)
            graph_list.append(np_graph)
    elif isinstance(graphs, nx.classes.graph.Graph):
        graph_list = nx.to_numpy_array(graphs)
    else:
        raise ValueError('Input needs to be either a list of networkx graphs or a networkx photonai_graph.')

    return graph_list


def networkx_to_sparse(graphs, fmt='csr'):
    """converts networkx graph to sparse graphs

        Parameters
        ---------
        graphs: list or single nx.classes.graph.Graph
            list of graphs or single to be converted to scipy sparse matrix/matrices
        fmt: str, default="csr"
            format of the scipy sparse matrix
    """
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            sparse_graph = nx.to_scipy_sparse_matrix(graph, format=fmt)
            graph_list.append(sparse_graph)
    elif isinstance(graphs, nx.classes.graph.Graph):
        graph_list = nx.to_scipy_sparse_matrix(graphs, format=fmt)
    else:
        raise ValueError('Input needs to be a list of networkx graphs or a networkx photonai_graph.')

    return graph_list


def networkx_to_dgl(graphs, node_attrs=None, edge_attrs=None):
    """convert networkx graphs to dgl graphs

        Parameters
        --------
        graphs: list,
            list or single networkx graphs to be converted
        node_attrs: list, default=None
            name of node attributes as list
        edge_attrs: list, default=None
            name of edge attributes as list
    """
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            if not nx.is_directed(graph):
                graph = graph.to_directed()
            if node_attrs or edge_attrs is None:
                g = dgl.from_networkx(graph)
            else:
                g = dgl.DGLGraph()
                g.from_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
            graph_list.append(g)
    elif isinstance(graphs, np.ndarray):
        graph_list = []
        for graph in range(graphs.shape[0]):
            if not nx.is_directed(graphs[graph]):
                graphs[graph] = graphs[graph].to_directed()
            if node_attrs or edge_attrs is None:
                g = dgl.from_networkx(graphs[graph])
            else:
                g = dgl.from_networkx(graphs[graph], node_attrs=node_attrs, edge_attrs=edge_attrs)
            graph_list.append(g)
    elif isinstance(graphs, nx.classes.graph.Graph):
        if not nx.is_directed(graphs):
            graphs = graphs.to_directed()
        g = dgl.from_networkx(graphs)
        graph_list = g
    else:
        raise ValueError('networkx_to_dgl only implemented for list or single networkx graph')

    return graph_list


def sparse_to_networkx(graphs):
    """convert scipy sparse matrices to networkx graphs

        Parameters
        ---------
        graphs:
            list of sparse matrices or single sparse matrix
    """
    if isinstance(graphs, list):
        graph_list = []
        for graph in graphs:
            nx_graph = nx.from_scipy_sparse_matrix(graph)
            graph_list.append(nx_graph)
    else:
        try:
            graph_list = nx.from_scipy_sparse_matrix(graphs)
        except Exception as e:
            print("Please check your input.\n"
                  "This function only takes a single sparese matrix or a list of sparse matrices.")
            raise e

    return graph_list


def dense_to_networkx(mtrx, adjacency_axis=0, feature_axis=1, feature_construction="features"):
    """converts an array or list of arrays to a list of networkx graphs

        Parameters
        ---------
        mtrx: list or np.ndarray
            numpy matrices to be converted
        adjacency_axis: int, default=0
            position of the adjacency matrices
        feature_axis: int, default=1
            position of the feature matrices
        feature_construction: str, default="features"
            method of node feature construction, for more information see
            get_dense_feature function
    """
    if isinstance(mtrx, list):
        graph_list = []

        for i in mtrx:
            networkx_graph = nx.from_numpy_matrix(A=i[:, :, adjacency_axis])
            features = get_dense_feature(i, adjacency_axis, feature_axis, aggregation=feature_construction)
            nx.set_node_attributes(networkx_graph, features, name="feat")
            graph_list.append(networkx_graph)

        mtrx_conv = graph_list

    # convert if Dense is an ndarray consisting of multiple arrays
    elif isinstance(mtrx, np.ndarray):
        # if dense is just a single array
        if mtrx.shape[0] == 1:
            mtrx_conv = nx.from_numpy_matrix(A=mtrx[0, :, :, adjacency_axis])
            features = get_dense_feature(mtrx, adjacency_axis, feature_axis, feature_construction)
            nx.set_node_attributes(mtrx_conv, features, name="feat")
        # if dense consists of multiple arrays
        else:
            graph_list = []
            # loop over each graph
            for i in range(mtrx.shape[0]):
                i_graph = mtrx[i, :, :, :]
                networkx_graph = nx.from_numpy_matrix(A=i_graph[:, :, adjacency_axis])
                features = get_dense_feature(i_graph, adjacency_axis, feature_axis, aggregation=feature_construction)
                nx.set_node_attributes(networkx_graph, features, name="feat")
                graph_list.append(networkx_graph)

            mtrx_conv = graph_list

    else:
        raise ValueError('input needs to be list of arrays or ndarray')

    return mtrx_conv


def get_dense_feature(matrix, adjacency_axis, feature_axis, aggregation="sum"):
    """returns the features for a networkx graph

        Parameters
        ---------
        matrix: np.matrix/np.ndarray
            feature matrix
        adjacency_axis: int
            position of the adjacency matrix
        feature_axis: int
            position of the feature matrix
        aggregation:
            method of feature construction, sum gives a row-wise sum,
            "mean" gives a row-wise mean, "node_degree" give a row-wise node-degree,
            features returns the entire row as the feature vector
    """
    if aggregation == "sum":
        features = np.sum(matrix[:, :, feature_axis], axis=1)
        features = features.tolist()
    elif aggregation == "mean":
        features = (np.sum(matrix[:, :, feature_axis], axis=1)) / matrix.shape[0]
        features = features.tolist()
    elif aggregation == "node_degree":
        features = np.count_nonzero(matrix[:, :, adjacency_axis], axis=1, keepdims=False)
        features = features.tolist()
    elif aggregation == "features":
        features = matrix[:, :, feature_axis]
        features = features.reshape((features.shape[0], -1))
        features = features.tolist()
    else:
        raise KeyError('Only sum, mean, node_degree and all features are supported')

    features = dict(enumerate(features, 0))

    # features = {str(key): value for key, value in features.items()}

    return features


def get_dense_edge_features(matrix, adjacency_axis, feature_axis):
    """returns the features for an edge label dictionary

        Parameters
        ---------
        matrix: np.matrix/np.ndarray
            feature matrix
        adjacency_axis: int
            position of the adjacency matrix
        feature_axis: int
            position of the feature matrix
    """
    edge_feat = {}
    for index, value in np.ndenumerate(matrix[:, :, adjacency_axis]):
        conn_key = (str(index[0]), str(index[1]))
        key_val = {conn_key: value}
        edge_feat.update(key_val)

    # edge_feat = dict(np.ndenumerate(matrix[:, :, adjacency_axis]))

    return edge_feat


def dense_to_sparse(graphs, m_type="coo_matrix", adjacency_axis=None, feature_axis=None):
    """converts numpy dense matrices to scipy sparse matrices

        Parameters
        ---------
        graphs: list or np.ndarray or np.matrix
            graphs in dense format which are to be converted
        m_type: str, default="coo_matrix"
            matrix format for the scipy sparse matrices
        adjacency_axis: int, default=None
            position of the adjacency matrix
        feature_axis: int, default=None
            position of the feature matrix

    """
    if m_type not in sparse_types:
        raise KeyError('Youor desired output format is not supported.\nPlease check your output format.')

    if isinstance(graphs, list):
        graph_list = []
        feature_list = []
        if np.ndim(graphs[0]) == 2:
            for graph in graphs:
                if np.ndim(graph) != 2:
                    raise ValueError('All graphs must have same shape.')
                graph_list.append(sparse_types[m_type](graph))
        elif np.ndim(graphs[0]) == 3:
            for graph in graphs:
                if np.ndim(graph) != 3:
                    raise ValueError('All graphs must have same shape.')
                sparse_graph = sparse_types[m_type](graph[:, :, adjacency_axis])
                sparse_features = sparse_types[m_type](graph[:, :, feature_axis])
                feature_list.append(sparse_features)
                graph_list.append(sparse_graph)
        else:
            raise ValueError('Unsupported graph shape.')

    elif isinstance(graphs, np.matrix) or isinstance(graphs, np.ndarray):
        graph_list = []
        feature_list = []
        if adjacency_axis is not None and feature_axis is not None:
            # if 4 dimensions you have subjects x values x values x adjacency/features
            if np.ndim(graphs) == 4:
                for i in range(graphs.shape[0]):
                    adjacency = sparse_types[m_type](graphs[i, :, :, adjacency_axis])
                    graph_list.append(adjacency)
                    feature = sparse_types[m_type](graphs[i, :, :, feature_axis])
                    feature_list.append(feature)
            elif np.ndim(graphs) == 3:
                graph_list = sparse_types[m_type](graphs[:, :, adjacency_axis])
                feature_list = sparse_types[m_type](graphs[:, :, feature_axis])
            else:
                raise ValueError("Matrix needs to have 4 or 3 dimensions when axis arguments are given.")

        else:
            if np.ndim(graphs) == 3:
                for i in range(graphs.shape[0]):
                    adjacency = sparse_types[m_type](graphs[i, :, :])
                    graph_list.append(adjacency)
            elif np.ndim(graphs) == 2:
                graph_list = sparse_types[m_type](graphs)
            else:
                raise ValueError("Matrix needs to have 3 or 2 dimension when no axis arguments are given.")

    else:
        raise ValueError("Input needs to be a numpy array or a list of arrays.")

    return graph_list, feature_list


def dense_to_dgl(graphs, adjacency_axis=None, feature_axis=None):
    """Converts dense matrices to dgl graphs

        Parameters
        ---------
        graphs: list, np.ndarray or np.matrix
            graphs represented in dense format
        adjacency_axis: int, default=None
            position of the adjacency matrix
        feature_axis: int, default=None
            position of the feature matrix
    """
    if adjacency_axis is None:
        raise NotImplementedError('dense to dgl not implemented without adjacency axis')
    else:
        sparse_mtrx = dense_to_sparse(graphs, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        graph_list = sparse_to_dgl(sparse_mtrx)
    return graph_list


def sparse_to_dense(graphs, features=None):
    """convert sparse matrices to numpy array

        Parameters
        ---------
        graphs: list or scipy matrix
            a list of scipy sparse matrices or a single sparse matrix
        features: list or scipy matrix, default=None
            if a feature matrix or a list of those is specified they are
            incorporated into the numpy array
    """
    def convert_matrix(current_graph_mtrx, current_features):
        """
        Helper function
        """
        current_graph_mtrx = np.reshape(current_graph_mtrx, (current_graph_mtrx.shape[0], current_graph_mtrx.shape[1], -1))
        current_features = current_features.toarray()
        current_features = np.reshape(current_features, (current_features.shape[0], current_features.shape[1], -1))
        return np.concatenate((current_graph_mtrx, current_features), axis=2)

    if features is not None:
        if isinstance(graphs, list):
            matrices = []
            for graph, feature in zip(graphs, features):
                graph_mtrx = graph.toarray()
                matrices.append(convert_matrix(graph_mtrx, feature))
        else:
            try:
                graph_mtrx = graphs.toarray()
                matrices = convert_matrix(graph_mtrx, features)
            except Exception as e:
                print('Could not convert matrices.'
                      'Your matrices need to be a list or a single sparse matrix.')
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

    matrices = np.asarray(matrices)

    return matrices


def sparse_to_dgl(graphs, adjacency_axis=0, feature_axis=1):
    """convert sparse matrices to dgl graphs

        Parameters
        ---------
        graphs: tuple, list, np.ndarray or np.matrix
            tuple consisting of two lists
        adjacency_axis: int, default=0
            position of the adjacency matrix
        feature_axis: int, default=1
            position of the feature matrix
    """
    if isinstance(graphs, tuple):
        graph_list = []
        for adj, feat in zip(graphs[adjacency_axis], graphs[feature_axis]):
            g = dgl.from_scipy(sp_mat=adj)
            graph_list.append(g)
    elif isinstance(graphs, list):
        graph_list = []
        for adj in graphs:
            g = dgl.from_scipy(sp_mat=adj)
            graph_list.append(g)
    elif isinstance(graphs, np.ndarray) or isinstance(graphs, np.matrix):
        if np.ndim(graphs) != 1:
            raise Exception('Input needs to be 1d array if it is a numpy array')
        graph_list = []
        for adj in range(graphs.shape[0]):
            g = dgl.from_scipy(sp_mat=graphs[adj])
            graph_list.append(g)

    else:
        raise Exception('Expected tuple, list or 1d-array as input.')

    return graph_list


def dgl_to_dense(graphs, in_fmt="csr"):
    """turns dgl graphs into dense matrices

        Parameters
        ---------
        graphs: list
            list of dgl graphs
        in_fmt: str, default="csr"
            format of the scipy sparse matrix used in the intermediary step
    """
    if not isinstance(graphs, list):
        raise Exception('Input graphs need to be in list format')

    sp_graphs = dgl_to_sparse(graphs, in_fmt)
    graph_list = sparse_to_dense(sp_graphs)
    return graph_list


def dgl_to_sparse(graphs, fmt="csr"):
    """turns dgl graphs into sparse matricesParameters

        Parameters
        ---------
        graphs: list
            list of dgl graphs
        fmt: str, default="csr"
            format of the scipy sparse matrix used in the intermediary step
    """
    if not isinstance(graphs, list):
        raise Exception("Input type needs to be a list")

    graph_list = []
    for graph in graphs:
        scp_graph = graph.adjacency_matrix_scipy(fmt=fmt)
        graph_list.append(scp_graph)
    return graph_list


def dgl_to_networkx(graphs, node_attrs=None, edge_attrs=None):
    """turns dgl graph into networkx graphs

        Parameters
        ---------
        graphs: list
            list of dgl graphs to be converted to networkx graphs
        node_attrs: list, default=None
            Node attributes as a list of strings
        edge_attrs: list, default=None
            Edge attributes as a list of strings
    """
    if node_attrs is None:
        node_attrs = ["feat"]
    if edge_attrs is None:
        edge_attrs = ["weight"]

    # check input type
    if not isinstance(graphs, list):
        raise Exception('Input graphs need to be in list format')

    graph_list = []
    for graph in graphs:
        nx_graph = dgl.DGLGraph.to_networkx(graph, node_attrs, edge_attrs)
        graph_list.append(nx_graph)
    return graph_list


def check_dgl(graphs, adjacency_axis=None, feature_axis=None):
    """Checks the input and converts it to dgl format

        Parameters
        ---------
        graphs: list, np.ndarray or np.matrix
            graphs to be converted
        adjacency_axis: int, default=None
            position of the adjacency matrix
        feature_axis: int, default=None
            position of the feature matrix
    """
    if not isinstance(graphs, np.ndarray) and not isinstance(graphs, np.matrix) and not isinstance(graphs, list):
        raise TypeError('can only handle np arrays or lists as input')

    if isinstance(graphs, list) or np.ndim(graphs) == 1:
        if isinstance(graphs[0], nx.classes.graph.Graph):
            dgl_graphs = networkx_to_dgl(graphs)
        elif isinstance((graphs[0]), dgl.DGLGraph):
            dgl_graphs = graphs
        else:
            try:
                dgl_graphs = sparse_to_dgl(graphs)
            except Exception as e:
                print('Can only handle ndarrays of networkx graphs or scipy matrices')
                raise e
    elif np.ndim(graphs) > 1:
        dgl_graphs = dense_to_dgl(graphs, adjacency_axis, feature_axis)
    else:
        raise ValueError('numpy matrix must have between one and four dimensions')

    return dgl_graphs


conversion_functions = {
    ("networkx", "dense"): networkx_to_dense,
    ("networkx", "sparse"): networkx_to_sparse,
    ("networkx", "dgl"): networkx_to_dgl,
    ("dense", "networkx"): dense_to_networkx,
    ("dense", "sparse"): dense_to_sparse,
    ("dense", "dgl"): dense_to_dgl,
    ("sparse", "networkx"): sparse_to_networkx,
    ("sparse", "dense"): sparse_to_dense,
    ("sparse", "dgl"): sparse_to_dgl,
    ("dgl", "networkx"): dgl_to_networkx,
    ("dgl", "dense"): dgl_to_dense,
    ("dgl", "sparse"): dgl_to_sparse
}


def convert_graphs(graphs, input_format="networkx", output_format="stellargraph"):
    """Convert graphs from one format to the other.

        Parameters
        ----------
        graphs : graphs
            list of graphs, or np.ndarray/np.matrix

        input_format : str, default="networkx"
            format of the graphs to be transformed

        output_format : str, default="stellargraph"
            desired output format of the graph(s)

        Returns
        -------
        list, np.ndarray, np.matrix
            The transformed matrices as a list

        Notes
        -----
        Output format is referenced by package name, written in lowercase letters

        """
    if input_format == output_format:
        warnings.warn('Graphs already in desired format.')

        return graphs

    if (input_format, output_format) not in conversion_functions.keys():
        raise TypeError('Your desired conversion is not supported.'
                        'Please check your in- and output format')

    trans_graphs = conversion_functions[(input_format, output_format)](graphs)

    return trans_graphs
