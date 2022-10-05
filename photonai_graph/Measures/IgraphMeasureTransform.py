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

import igraph
from tqdm.contrib.concurrent import thread_map
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
import pandas as pd
import numpy as np
import json
import os

from photonai_graph.GraphConversions import dense_to_networkx
from photonai_graph.Measures.AbstractMeasureTransform import AbstractMeasureTransform


class IgraphMeasureTransform(AbstractMeasureTransform):
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
        super(IgraphMeasureTransform, self).__init__(graph_functions=graph_functions)
        if self.graph_functions is None:
            self.graph_functions = {"degree": {}, "eigenvector_centrality": {}}
        self.n_processes = n_processes
        self.adjacency_axis = adjacency_axis

        self.logs = logs
        if not logs:
            self.logs = os.getcwd()

    def fit(self, X, y):
        return self

    def _inner_transform(self, X):
        x_transformed = []

        graphs = X

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        measure_json = os.path.join(base_folder, 'IgraphMeasures.json')
        with open(measure_json, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        if self.n_processes > 1:
            pfn = partial(self._compute_graph_metrics, graph_functions=self.graph_functions, measure_j=measure_j)
            x_transformed = thread_map(pfn, graphs, max_workers=self.n_processes)
        else:
            for graph in graphs:
                measure_list_graph = self._compute_graph_metrics(graph, self.graph_functions, measure_j)
                x_transformed.append(measure_list_graph)

        return self._shared_inner_transform(x_transformed=x_transformed)

    def transform(self, X):
        if not isinstance(X, (list, np.ndarray, np.matrix)):
            raise TypeError('Input needs to list/ndarray of Igraph objects')
        if not isinstance(X[0], igraph.Graph):
            raise TypeError('Input needs to be list/ndarray or Igraph objects')

        x_transformed = self._inner_transform(X)
        return self._shared_transform(x_transformed=x_transformed)

    def _compute_graph_metrics(self, graph, graph_functions, measure_j):
        measure_list_graph = []
        for key, value in graph_functions.items():
            measure_list = list()

            if key not in measure_j:
                raise ValueError(f"Measure function {key} not found")

            measure = measure_j[key]

            # call function
            results = getattr(graph, measure["path"].split(".")[-1])(**value)
            measure_list = self.handle_outputs(results, measure_list)

            if "compute_average" in measure.keys() and measure['compute_average']:
                measure_list_graph.append([np.mean(measure_list)])
            else:
                measure_list_graph.append(measure_list)
        return measure_list_graph

    def get_measure_info(self):
        pass

    def extract_measures(self, x_graphs_in, path="", ids=None):
        x_graphs = x_graphs_in.copy()
        if ids is None:
            raise ValueError('No id provided')
        self._shared_extraction(x_graphs, ids, path)
