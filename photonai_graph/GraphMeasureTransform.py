"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for extracting photonai_graph measures that can then be
used for further machine learning analyses

Version
-------
Created:        09-09-2019
Last updated:   02-06-2020


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

# TODO: make error messages for possible errors
# TODO: make documentation for every single method

import networkx
from tqdm.contrib.concurrent import thread_map
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
import pandas as pd
import numpy as np
import json
import os

from photonai_graph.GraphConversions import dense_to_networkx


class GraphMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 graph_functions: dict = None,
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
        if graph_functions is None:
            graph_functions = {"global_efficiency": {}, "average_node_connectivity": {}}
        self.graph_functions = graph_functions
        self.adjacency_axis = adjacency_axis

        self.logs = logs
        if not logs:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def _inner_transform(self, X):
        x_transformed = []

        if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
            graphs = dense_to_networkx(X, adjacency_axis=self.adjacency_axis)
        elif isinstance(X, list) and min([isinstance(g, nx.classes.graph.Graph) for g in X]):
            graphs = X
        else:
            raise TypeError("Input needs to be list of networkx graphs or numpy array.")

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        measure_json = os.path.join(base_folder, 'photonai_graph/GraphMeasures.json')
        with open(measure_json, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        if self.n_processes > 1:
            pfn = partial(self._compute_graph_metrics, graph_functions=self.graph_functions, measure_j=measure_j)
            x_transformed = thread_map(pfn, graphs, max_workers=self.n_processes)
        else:
            for graph in graphs:
                measure_list_graph = self._compute_graph_metrics(graph, self.graph_functions, measure_j)
                x_transformed.append(measure_list_graph)

        for c_measure in range(len(self.graph_functions)):
            expected_values = max([len(graph[c_measure]) for graph in x_transformed])
            for graph in x_transformed:
                if len(graph[c_measure]) < expected_values:
                    graph[c_measure] = [np.NAN] * expected_values

        return x_transformed

    def transform(self, X):
        X_transformed = self._inner_transform(X)

        for graph_idx in range(len(X_transformed)):
            g_m = list()
            for measure in X_transformed[graph_idx]:
                g_m.extend(measure)
            X_transformed[graph_idx] = g_m

        X_transformed = np.asarray(X_transformed)

        return X_transformed

    def _compute_graph_metrics(self, graph, graph_functions, measure_j):
        measure_list_graph = []
        for key, value in graph_functions.items():
            measure_list = list()

            if key not in measure_j:
                raise ValueError(f"Measure functino {key} not found")

            measure = measure_j[key]
            # remove self loops if not allowed
            if not measure['self_loops_allowed']:
                graph.remove_edges_from(networkx.selfloop_edges(graph))
            # make photonai_graph directed or undirected depending on what is needed
            if measure['Undirected']:
                graph.to_undirected()
            elif not measure['Undirected']:
                graph.to_directed()

            # call function
            results = getattr(networkx, measure["path"].split(".")[-1])(graph, **value)
            measure_list = self.handle_outputs(results, measure_list)

            if "compute_average" in measure.keys() and measure['compute_average']:
                measure_list_graph.append([np.mean(measure_list)])
            else:
                measure_list_graph.append(measure_list)
        return measure_list_graph

    @staticmethod
    def handle_outputs(results, measure_list):
        # handle results
        if isinstance(results, dict):
            for rskey, rsval in results.items():
                GraphMeasureTransform.handle_outputs(rsval, measure_list)
            return measure_list

        if isinstance(results, list):
            measure_list.extend(results)
            return measure_list

        # currently only networkx functions return tuples
        # The second return value can be discarded in these functions
        if isinstance(results, tuple):
            for result in results:
                GraphMeasureTransform.handle_outputs(result, measure_list)
            return measure_list

        if isinstance(results, nx.Graph):
            return measure_list

        measure_list.append(results)
        return measure_list

    def get_measure_info(self):
        pass

    def extract_measures(self, x_graphs_in, path="", ids=None):
        x_graphs = x_graphs_in.copy()
        if ids is None:
            raise ValueError('No id provided')
        if isinstance(x_graphs, np.ndarray):
            # [..., 0] because we are discarding the feature axis
            x_graphs = [nx.from_numpy_array(x_graphs[cid][..., 0]) for cid in ids]
        else:
            x_graphs = [x_graphs[cid] for cid in ids]
        X_transformed = self._inner_transform(x_graphs)

        measurements = []
        for graph, gid in zip(X_transformed, ids):
            for measurement_id, result in enumerate(graph):
                for res in result:
                    current_measurement = [gid, list(self.graph_functions.keys())[measurement_id], res]
                    measurements.append(current_measurement)

        df = pd.DataFrame(measurements)

        col_names = ["graph_id", "measure", "value"]

        df.to_csv(path_or_buf=path, header=col_names, index=None)
