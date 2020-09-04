"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A collection of functions for converting photonai_graph structure formats, Visualize the different graphs,
and even create some random data to run some tests

Version
-------
Created:        15-08-2019
Last updated:   04-09-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

from networkx.convert_matrix import from_numpy_matrix
from networkx.drawing.nx_pylab import draw
import networkx.drawing as drawx
from networkx.algorithms import asteroidal
import networkx as nx
import numpy as np
import pydot
from scipy import stats
from scipy import sparse
import os
import matplotlib.pyplot as plt
from photonai_graph.GraphConversions import save_networkx_to_file, networkx_to_dense, networkx_to_sparse, \
    networkx_to_stellargraph
from photonai_graph.GraphConversions import dense_to_networkx as dnx, dense_to_stellargraph, dense_to_sparse
from photonai_graph.GraphConversions import sparse_to_networkx, sparse_to_dense, sparse_to_stellargraph
from photonai_graph.GraphConversions import stellargraph_to_networkx, stellargraph_to_dense, stellargraph_to_sparse
import warnings


def DenseToNetworkx(X, adjacency_axis=0, feature_axis=1, feature_construction="collapse"):
    # todo: I have renamed this function. Use dense_to_networkx in the future
    warnings.warn("This function is renamed to dense_to_networkx", DeprecationWarning)
    return dense_to_networkx(X, adjacency_axis, feature_axis, feature_construction)


def dense_to_networkx(mtrx, adjacency_axis=0, feature_axis=1, feature_construction="collapse"):
    """converts an array or list of arrays to a list of networkx graphs"""
    if isinstance(mtrx, list):
        graph_list = []

        for i in mtrx:
            networkx_graph = from_numpy_matrix(A=i[:, :, adjacency_axis])
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

        mtrx_conv = graph_list

    # convert if Dense is just a single ndarray
    elif isinstance(mtrx, nx.classes.graph.Graph):
        mtrx_conv = from_numpy_matrix(A=mtrx[:, :, adjacency_axis])

    # convert if Dense is an ndarray consisting of multiple arrays
    elif isinstance(mtrx, np.ndarray):
        graph_list = []

        for i in range(mtrx.shape[0]):
            networkx_graph = from_numpy_matrix(A=mtrx[i, :, :, adjacency_axis])
            if feature_construction == "collapse":
                features = np.sum(mtrx[i, :, :, feature_axis], axis=1)
                features = features.reshape((features.shape[0], -1))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features, name="collapsed_weight")
            elif feature_construction == "collapse2":
                sum_features = np.sum(mtrx[i, :, :, feature_axis], axis=1)
                var_features = np.var(mtrx[i, :, :, feature_axis], axis=1)
                features = np.column_stack((sum_features, var_features))
                features = dict(enumerate(features, 0))
                nx.set_node_attributes(networkx_graph, features, name="collapsed_weight")
            graph_list.append(networkx_graph)

        mtrx_conv = graph_list
    else:
        raise ValueError("X has unsupported format. Please check input")

    return mtrx_conv


def draw_connectogram(graph, edge_rad=None, colorscheme=None, nodesize=None,
                      node_shape='o', weight=None, path=None):
    """This functions draws a connectogram, from a graph.

    Parameters
    ----------
    graph: nx.class.graph.Graph
        input graph, a single networkx graph
    edge_rad:
        edge radius, controlling the curvature of the drawn edges
    colorscheme:
        colormap for drawing the connectogram
    nodesize: int
        controls size of the drawn nodes
    node_shape: str, default='o'
        shape of the drawn nodes
    weight: float
        threshold below which edges are coloured differently than above
    path: str, default=None
        path where to save the plots as string, if no path is declared, plots are not saved.
        Path needs to be the full path including file name and ending, unlike in draw_connectograms
    """

    pos = nx.circular_layout(graph)
    if colorscheme is not None:
        nx.draw_networkx_nodes(graph, pos, node_size=nodesize, node_color=range(nx.number_of_nodes(graph)),
                               node_shape=node_shape, cmap=plt.get_cmap(colorscheme))
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=nodesize, node_shape=node_shape, cmap=plt.get_cmap(colorscheme))

    if weight is not None:
        elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > weight]
        esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= weight]
        nx.draw_networkx_edges(graph, pos, edgelist=elarge,
                               connectionstyle=edge_rad)
        nx.draw_networkx_edges(graph, pos, edgelist=esmall,
                               alpha=0.5, edge_color='b', style='dashed', connectionstyle=edge_rad)

    else:
        nx.draw_networkx_edges(graph, pos, connectionstyle=edge_rad)

    plt.show()

    if path is not None:
        plt.savefig(path)


def draw_connectograms(graphs, curved_edge=False, colorscheme=None,
                       nodesize=None, node_shape='o', weight=None,
                       path=None, ids=None, out_format=None):
    """This function draws multiple connectograms, from graph lists.

    Parameters
    ----------
    graphs
        input graphs, a list of networkx graphs or a single networkx graph
    curved_edge: bool, default=False
        whether to draw straight or curved edges
    colorscheme:
        colormap for drawing the connectogram
    nodesize: int
        controls size of the drawn nodes
    node_shape: str, default='o'
        shape of the drawn nodes
    weight: float
        threshold below which edges are coloured differently than above
    path: str, default=None
        path where to save the plots as string, if no path is declared, plots are not saved
    ids: list, default=None
        list of ids, after which to name the plots
    out_format: str, default=None
        output format for the graphs, as a string
    """
    if isinstance(graphs, list):
        if ids is not None:
            if len(ids) == len(graphs):
                for graph, ID in zip(graphs, ids):
                    if None in [path, out_format]:
                        raise Exception('To save graphs, declare a path and an output format.')
                    save_path = os.path.join(path, ID, out_format)
                    draw_connectogram(graph, curved_edge, colorscheme, path=save_path)
            else:
                raise Exception('Number of IDs must match number of graphs.')
        # if no IDs are provided graphs are just numbered
        else:
            counter = 0
            if None in [path, out_format]:
                for graph in graphs:
                    draw_connectogram(graph, curved_edge, colorscheme, nodesize, node_shape, weight)
                    counter += 1
            else:
                for graph in graphs:
                    save_path = os.path.join(path, str(counter), out_format)
                    draw_connectogram(graph, curved_edge, colorscheme, nodesize, node_shape, weight, path=save_path)
                    counter += 1

    # if the it is only a single graph
    elif isinstance(graphs, nx.classes.graph.Graph):
        draw_connectogram(graphs, curved_edge, colorscheme, path=path)

    # input should be list or single graph
    else:
        raise Exception('Input needs to be a single networkx graph or a list of those.')


def draw_connectivity_matrix(matrix, colorbar=False, colorscheme="viridis", adjacency_axis=None):
    """Draw connectivity matrix.

        Parameters
        ----------
        matrix : numpy.ndarray, numpy.matrix or a list of those
        the input matrix or matrices from which to draw the connectivity matrix
            
        colorbar : boolean, default=False
            Whether to use a colorbar in the drawn plot

        colorscheme: str, default="viridis"
            colorscheme for plotting the connectivity matrix

        adjacency_axis : int, default=None
        position of the the adjacency axis, if specified the array is assumed to
        have an additional axis where the matrix is stored.

        Notes
        -----
        If new nodes are added with features, and any of the old nodes
        do not have some of the feature fields, those fields are filled
        by initializers defined with ``set_n_initializer`` (default filling
        with zeros).

        Examples
        --------
        >>> g = get_random_connectivity_data()
        >>> draw_connectivity_matrix(adjacency_axis=0)
               
        """
    # check input format
    if isinstance(matrix, np.ndarray) or isinstance(matrix, np.matrix):
        if adjacency_axis is not None:
            if np.ndim(matrix) == 4:
                for i in range(matrix.shape[0]):
                    plt.imshow(matrix[i, :, :, adjacency_axis], cmap=plt.get_cmap(colorscheme))
                    if colorbar:
                        plt.colorbar()
                    plt.show()
            elif np.ndim(matrix) == 3:
                plt.imshow(matrix[:, :, adjacency_axis], cmap=plt.get_cmap(colorscheme))
                if colorbar:
                    plt.colorbar()
                plt.show()
            else:
                raise Exception('Matrix dimension might not be specified correctly.')
        else:
            if np.ndim(matrix) == 4:
                raise Exception('You have 4 dimension, please specify axis to plot')
            elif np.ndim(matrix) == 3:
                for i in range(matrix.shape[0]):
                    plt.imshow(matrix[i, :, :])
                    if colorbar:
                        plt.colorbar()
                    plt.show()
            elif np.ndim(matrix) == 2:
                plt.imshow(matrix)
                if colorbar:
                    plt.colorbar()
                plt.show()
            else:
                raise Exception('Matrix dimension might not be specified correctly.')
    elif isinstance(matrix, list):
        if isinstance(matrix[0], np.ndarray) or isinstance(matrix[0], np.matrix):
            for single_matrix in matrix:
                if adjacency_axis is not None:
                    plt.imshow(single_matrix[:, :, adjacency_axis])
                    if colorbar:
                        plt.colorbar()
                    plt.show()
        elif isinstance(matrix[0], sparse.spmatrix) \
                or isinstance(matrix[0], sparse.bsr_matrix) \
                or isinstance(matrix[0], sparse.lil_matrix) \
                or isinstance(matrix[0], sparse.csc_matrix) \
                or isinstance(matrix[0], sparse.coo_matrix) \
                or isinstance(matrix[0], sparse.csr_matrix) \
                or isinstance(matrix[0], sparse.dok_matrix) \
                or isinstance(matrix[0], sparse.dia_matrix):
            for single_matrix in matrix:
                plt.spy(matrix[single_matrix])
        else:
            raise TypeError('List elements need to be numpy arrays/matrices or scipy sparse matrices')
    else:
        raise TypeError('draw_connectivity_matrix only takes numpy arrays, matrices or lists as input.')


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

        Examples
        --------
        >>> g = get_random_connectivity_data()
        >>> draw_connectivity_matrix(g, adjacency_axis=0)
               
        """
    # check input format
    if input_format == "networkx":
        if output_format == "networkx":
            warnings.warn('Graphs already in networkx format.')
            trans_graphs = graphs
        elif output_format == "dense":
            trans_graphs = networkx_to_dense(graphs)
        elif output_format == "sparse":
            trans_graphs = networkx_to_sparse(graphs)
        elif output_format == "stellargraph":
            trans_graphs = networkx_to_stellargraph(graphs)
        else:
            raise KeyError('Your specified output format is not supported.'
                           'Please check your output format.')
    elif input_format == "dense":
        if output_format == "networkx":
            trans_graphs = dnx(graphs)  # dense_to_networkx is redefined above, so we imported it as dnx
        elif output_format == "dense":
            raise Exception('Graphs already in dense format.')
        elif output_format == "sparse":
            trans_graphs = dense_to_sparse(graphs)
        elif output_format == "stellargraph":
            trans_graphs = dense_to_stellargraph(graphs)
        else:
            raise KeyError('Your specified output format is not supported.'
                           'Please check your output format.')
    elif input_format == "sparse":
        if output_format == "networkx":
            trans_graphs = sparse_to_networkx(graphs)
        elif output_format == "dense":
            trans_graphs = sparse_to_dense(graphs)
        elif output_format == "sparse":
            raise Exception('Graphs already in sparse format.')
        elif output_format == "stellargraph":
            trans_graphs = sparse_to_stellargraph(graphs)
        else:
            raise KeyError('Your specified output format is not supported.'
                           'Please check your output format.')
    elif input_format == "stellargraph":
        if output_format == "networkx":
            trans_graphs = stellargraph_to_networkx(graphs)
        elif output_format == "dense":
            trans_graphs = stellargraph_to_dense(graphs)
        elif output_format == "sparse":
            trans_graphs = stellargraph_to_sparse(graphs)
        elif output_format == "stellargraph":
            raise Exception('Graphs already in stellargraph format.')
        else:
            raise KeyError('Your specified output format is not supported.'
                           'Please check your output format.')
    else:
        raise KeyError('Your specified input format is not supported.'
                       'Please check your input format.')

    return trans_graphs


def get_random_connectivity_data(out_type="dense",
                                 number_of_nodes=114,
                                 number_of_individuals=10,
                                 number_of_modalities=2):
    """generate random connectivity matrices for testing and debugging

        Parameters
        ----------
        out_type: str, default="dense"
            output type for connectivity data, default="dense"
        number_of_nodes: int, default=114
            number of nodes in the matrix/graph
        number_of_individuals: int, default=10
            number of individual graphs/matrices
        number_of_modalities: int, default=2
            number of modalities as per matrix/graph
    """
    if out_type == "dense":
        random_matrices = np.random.rand(number_of_individuals, number_of_nodes, number_of_nodes, number_of_modalities)
    elif out_type == "sparse":
        random_matrices = []
        for i in range(number_of_individuals):
            modality_list = []
            for m in range(number_of_modalities):
                random_matrix = sparse.random(number_of_nodes, number_of_nodes, density=0.1)
                modality_list.append(random_matrix)
            random_matrices.append(modality_list)
    else:
        raise NotImplementedError("Only dense and sparse matrices are supported as output type.")

    return random_matrices


def get_random_labels(l_type="classification", number_of_labels=10):
    """get random labels for testing and debugging functions.

        Parameters
        ----------
        l_type : str, default="classification"
            controls the type labels. "classification" outputs binary labels 0 and 1, "regression" outputs random float.
            
        number_of_labels : int, default=10
            number of labels to generate

        Returns
        -------
        np.ndarray
            The labels as np.ndarray


        Notes
        -----
        If used in conjunction with get_random_connectivity_data number_of_labels
        should match number_of_individuals.

        Examples
        --------
        >>> labels = get_random_labels()
          
        """

    if l_type == "classification" or l_type == "Classification":
        y = np.random.rand(number_of_labels)
        y[y > 0.5] = 1
        y[y < 0.5] = 0

    elif l_type == "regression" or l_type == "Regression":
        y = np.random.rand(number_of_labels)

    else:
        # todo: If we simply print this message, the execution will not be aborted.
        # todo: In this case y is not defined in the return statement below.
        raise ValueError('random labels only implemented for classification and regression. Please check your spelling')

    return y


def save_graphs(graphs, path="", input_format="networkx", output_format="dot", ids=None):
    """save graphs to file.

        Parameters
        ----------
        graphs : 
            a list or a np.ndarray of graphs to be saved

        input_format : str, default="networkx"
            format of the graphs to be saved

        path : str, default=""
            path where to save the graphs
            
        output_format : str, default="dot"
            the output format in which to save the graphs

        ids :
            a list containing the ids of the graphs. Must have same length as the graph list or np.ndarray.


        Examples
        --------
        >>> g1, g2 = nx.line_graph(10), nx.line_graph(7)
        >>> graphs = [g1, g2]
        >>> save_graphs(graphs, path="path/to/your/data/")      
        
        """
    # check input format
    if input_format == "networkx":
        save_networkx_to_file(graphs, path, output_format=output_format, ids=ids)
    else:
        raise Exception("Your desired output format is not supported yet.")


def VisualizeNetworkx(graphs):
    # todo: I have moved this function
    warnings.warn("This function is renamed to visualize_networkx", DeprecationWarning)
    return visualize_networkx(graphs)


def visualize_networkx(graphs, layout=nx.spring_layout, colorscheme="Blues"):
    """Visualize a networkx graph or graphs using networkx built-in visualization.

        Parameters
        ----------
        graphs : 
            a list or of networkx graphs or a single networkx graph
        layout :
            layout of the graph, default is spring layout
        colorscheme:
            colormap for the nodes, default is Blues


        Examples
        --------
        >>> g1, g2 = nx.line_graph(10), nx.line_graph(7)
        >>> graphs = [g1, g2]
        >>> visualize_networkx(graphs)
       
        
        """
    # check format in which graphs are presented or ordered
    if isinstance(graphs, list):
        for graph in graphs:
            draw(graph, pos=layout(graph), node_color=range(nx.number_of_nodes(graph)), cmap=plt.get_cmap(colorscheme))
            plt.show()
    elif isinstance(graphs, nx.classes.graph.Graph):
        draw(graphs, pos=layout(graphs), node_color=range(nx.number_of_nodes(graphs)), cmap=plt.get_cmap(colorscheme))
        plt.show()
    else:
        raise ValueError("graphs has unexpected format")


def individual_ztransform(matrx, adjacency_axis=0):
    """applies a z-score transformation individually to each connectivity matrix in an array

        Parameters
        ----------
        matrx : np.ndarray
            a list or of networkx graphs or a single networkx graph

        adjacency_axis: int, default=0
            the position of the adjacency matrix

        Returns
        -------
        np.ndarray
            The z-score transformed matrices as an array


        Examples
        --------
        >>> X = get_random_connectivity_data()
        >>> X_transformed = individual_ztransform(X)
               
        """
    # check dimensions
    transformed_matrices = []
    if np.ndim(matrx) == 3:
        for i in range(matrx.shape[0]):
            matrix = matrx[i, :, :].copy()
            matrix = stats.zscore(matrix)
            transformed_matrices.append(matrix)
    elif np.ndim(matrx) == 4:
        for i in matrx.shape[0]:
            matrix = matrx[i, :, :, adjacency_axis].copy()
            matrix = stats.zscore(matrix)
            transformed_matrices.append(matrix)

    transformed_matrices = np.asarray(transformed_matrices)

    return transformed_matrices


def individual_fishertransform(matrx, adjacency_axis=0):
    """applies a fisher transformation individually to each connectivity matrix in an array

        Parameters
        ----------
        matrx : np.ndarray
            a list or of networkx graphs or a single networkx graph
        adjacency_axis: int, default=0
            the position of the adjacency matrix

        Returns
        -------
        np.ndarray
            The fisher transformed matrices as an array


        Examples
        --------
        >>> X = get_random_connectivity_data()
        >>> X_transformed = individual_fishertransform(X)

        """
    # check dimensions
    transformed_matrices = []
    if np.ndim(matrx) == 3:
        for i in range(matrx.shape[0]):
            matrix = matrx[i, :, :].copy()
            matrix = np.arctanh(matrix)
            transformed_matrices.append(matrix)
    elif np.ndim(matrx) == 4:
        for i in matrx.shape[0]:
            matrix = matrx[i, :, :, adjacency_axis].copy()
            matrix = np.arctanh(matrix)
            transformed_matrices.append(matrix)

    transformed_matrices = np.asarray(transformed_matrices)

    return transformed_matrices


def pydot_to_nx(graphs):
    """transforms pydot graphs to networkx graphs

        Parameters
        ----------
        graphs : list or pydot.Dot
        a list of pydot graphs or a single pydot graph


        Returns
        -------
        list or pydot.Dot
            The input graph or graphs in networkx format

        """
    if isinstance(graphs, list):
        a_graphs = []
        for graph in graphs:
            a_graph = drawx.nx_pydot.from_pydot(graph)
            a_graphs.append(a_graph)

    elif isinstance(graphs, pydot.Dot):
        a_graphs = drawx.nx_pydot.from_pydot(graphs)

    else:
        raise TypeError('The input needs to be list of pydot files or a single pydot file. Please check your inputs.')

    return a_graphs


def check_asteroidal(graph, return_boolean=True):
    """checks whether a graph or a list graphs is asteroidal

        Parameters
        ----------
        graph : list or nx.classes.graph.Graph
            a list of networkx graphs or a single networkx graph

        return_boolean : boolean, default=True
            whether to return a True/False statement about the graphs or a list of asteroidal triples per graph


        Returns
        -------
        list or boolean
            A list of True/False values or a list of asteroidal triples, or a single boolean value
            
        """
    # checks for asteroidal triples in the photonai_graph or in a list of networkx graphs
    if return_boolean:
        if isinstance(graph, list):
            graph_answer = []
            for i in graph:
                answer = asteroidal.is_at_free(i)
                graph_answer.append(answer)
        if isinstance(graph, nx.classes.graph.Graph):
            graph_answer = asteroidal.is_at_free(graph)
        else:
            raise ValueError('Your input is not a networkx photonai_graph or a list of networkx graphs. '
                             'Please check your inputs.')

    else:
        if isinstance(graph, list):
            graph_answer = []
            for i in graph:
                answer = asteroidal.find_asteroidal_triple(i)
                graph_answer.append(answer)
        if isinstance(graph, nx.classes.graph.Graph):
            graph_answer = asteroidal.find_asteroidal_triple(graph)
        else:
            raise ValueError('Your input is not a networkx photonai_graph or a list of networkx graphs. '
                             'Please check your inputs.')

    return graph_answer
