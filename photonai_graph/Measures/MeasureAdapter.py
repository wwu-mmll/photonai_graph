from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
import igraph
import numpy as np
import os

from photonai_graph.GraphConversions import dense_to_networkx, dense_to_igraph
from photonai_graph.util import NetworkxGraphWrapper


class GraphMeasureAdapter(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 output: str = None,
                 adjacency_axis: int = 0,
                 logs: str = None,
                 n_processes: int = 1):
        """The GraphMeasureTransform class is a class for extracting graph measures from graphs.
        A range of networkx graph measures is available and can be used in a PHOTON pipeline or extracted and
        written to a csv file for further use.

        Parameters
        ----------
        graph_functions: dict,default=None
            a dict of graph functions to calculate with keys as the networkx function name and a dict of the arguments
            as the value. In this second dictionary, the keys are the functions arguments and values are the desired
            values for the argument.
        adjacency_axis: int,default=0
            Channel index for adjacency
        logs: str,default=None
            path to the log data
        n_processes: str,default=None
            Number of processes for multi processing

        Examples
        --------
        ```python
        measuretrans = GraphMeasureTransform(graph_functions={"large_clique_size": {},
                                                              "global_efficiency": {},
                                                              "overall_reciprocity": {},
                                                              "local_efficiency": {}})
        ```
        """
        self.n_processes = n_processes
        self.output = output
        self.adjacency_axis = adjacency_axis

        self.logs = logs
        if not logs:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def _inner_transform(self, X):

        if self.output == 'igraph':
            if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
                graphs = dense_to_igraph(X, adjacency_axis=self.adjacency_axis)
            elif isinstance(X, list) and min([isinstance(g, igraph.Graph) for g in X]):
                graphs = X
            else:
                raise TypeError("Input needs to a list of igraph graphs or numpy array")

        elif self.output == 'networkx':
            if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
                graphs = dense_to_networkx(X, adjacency_axis=self.adjacency_axis)
            elif isinstance(X, list) and min([isinstance(g, nx.Graph) for g in X]):
                graphs = X
            else:
                raise TypeError("Input has to be a list of networkx graphs or numpy array.")

            graphs = [NetworkxGraphWrapper(g) for g in graphs]

        else:
            raise NotImplementedError('Can only convert to networkx or igraph.'
                                      'Please ensure spelling is correct.')

        return graphs

    def transform(self, X):

        X_transformed = self._inner_transform(X)

        X_transformed = np.asarray(X_transformed)

        return X_transformed
