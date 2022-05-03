import networkx as nx
import numpy as np
import warnings
from photonai_graph.util import assert_imported

try:
    import dgl
    import torch
except ImportError:  # pragma: no cover
    pass


def dense_to_networkx(graphs: np.ndarray, adjacency_axis: int = None, feature_axis=None):
    if adjacency_axis is None:
        warnings.warn("No adjacency passed. Guessing that adjacency is in first channel...")
        adjacency_axis = 0
    nx_graphs = [nx.from_numpy_array(graphs[i, ..., adjacency_axis]) for i in range(graphs.shape[0])]
    if feature_axis is not None:
        raise NotImplementedError("This feature is not implemented yet.")
    return nx_graphs


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
    assert_imported(["dgl"])

    if adjacency_axis is None:
        raise NotImplementedError('dense to dgl not implemented without adjacency axis')
    if not isinstance(graphs, np.ndarray):
        raise ValueError(f"Expected np.ndarray as input, got {type(graphs)}")

    graph_list = []
    for graph in range(graphs.shape[0]):
        src, dst = np.nonzero(graphs[graph, :, :, adjacency_axis])
        g = dgl.graph((src, dst), num_nodes=graphs[graph, ...].shape[1])
        feat = torch.tensor(graphs[graph, :, :, feature_axis])
        g.ndata['feat'] = feat
        graph_list.append(g)
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
        current_graph_mtrx = np.reshape(current_graph_mtrx,
                                        (current_graph_mtrx.shape[0], current_graph_mtrx.shape[1], -1))
        current_features = current_features.toarray()
        current_features = np.reshape(current_features, (current_features.shape[0], current_features.shape[1], -1))
        return np.concatenate((current_graph_mtrx, current_features), axis=2)

    if isinstance(graphs, list):
        matrices = []
        for idx, graph in enumerate(graphs):
            graph_mtrx = graph.toarray()
            if features is not None:
                matrices.append(convert_matrix(graph_mtrx, features[idx]))
            else:
                matrices.append(graph_mtrx)
    else:
        try:
            graph_mtrx = graphs.toarray()
            if features is not None:
                matrices = convert_matrix(graph_mtrx, features)
        except Exception as e:
            print('Could not convert matrices.'
                  'Your matrices need to be a list or a single sparse matrix.')
            raise e

    matrices = np.asarray(matrices)
    return matrices


def dgl_to_dense(graphs, in_fmt="csr"):
    """turns dgl graphs into dense matrices

        Parameters
        ---------
        graphs: list
            list of dgl graphs
        in_fmt: str, default="csr"
            format of the scipy sparse matrix used in the intermediary step
    """
    assert_imported(["dgl"])

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
    assert_imported(["dgl"])

    if not isinstance(graphs, list):
        raise Exception("Input type needs to be a list")

    graph_list = []
    for graph in graphs:
        scp_graph = graph.adjacency_matrix_scipy(fmt=fmt)
        graph_list.append(scp_graph)
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
    assert_imported(["dgl"])

    if not isinstance(graphs, np.ndarray) and not isinstance(graphs, np.matrix) and not isinstance(graphs, list):
        raise TypeError('can only handle np arrays or lists as input')

    if isinstance(graphs, list) or np.ndim(graphs) == 1:
        if not isinstance((graphs[0]), dgl.DGLGraph):
            raise ValueError("Expected list of dglGraph or dense matrix")
        dgl_graphs = graphs
    elif np.ndim(graphs) > 1:
        dgl_graphs = dense_to_dgl(graphs, adjacency_axis, feature_axis)
    else:
        raise ValueError('numpy matrix must have one or four dimensions')

    return dgl_graphs
