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
import dask
from dask.diagnostics import ProgressBar
from sklearn.base import BaseEstimator, TransformerMixin
from photonai_graph.GraphConversions import dense_to_networkx
import pandas as pd
import numpy as np
import json
import os


class GraphMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 graph_functions: dict = None,
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

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        X_transformed = []

        if isinstance(X, np.ndarray) or isinstance(X, np.matrix):
            graphs = dense_to_networkx(X, adjacency_axis=0)
        elif isinstance(X, list):
            graphs = X
        else:
            raise TypeError("Input needs to be list of networkx graphs or numpy array.")

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        measure_json = os.path.join(base_folder, 'photonai_graph/GraphMeasures.json')
        with open(measure_json, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        # compute graph metrics for first graph
        # ToDo: will only work in case first graph in list is not empty
        # ToDo: write unit test for parallelization
        measures_first_graph = self._compute_graph_metrics(graphs[0], self.graph_functions, measure_j)
        n_measures_to_expect = len(measures_first_graph)

        if self.n_processes > 1:
            task_list = []
            for graph in graphs:
                tmp = dask.delayed(self._compute_graph_metrics)(graph, self.graph_functions, measure_j, n_measures=None)
                task_list.append(tmp)
            with ProgressBar():
                X_transformed = list(dask.compute(*task_list, num_workers=self.n_processes, scheduler='threads'))
        else:

            for graph in graphs:
                measure_list_graph = self._compute_graph_metrics(graph, self.graph_functions, measure_j,
                                                                 n_measures_to_expect)
                X_transformed.append(measure_list_graph)

        X_transformed = np.asarray(X_transformed)

        return X_transformed

    @staticmethod
    def _compute_graph_metrics(graph, graph_functions, measure_j, n_measures=None):
        if networkx.classes.function.is_empty(graph):
            print("Graph is empty")
            if n_measures is None:
                return None
            else:
                return [np.nan] * n_measures
        measure_list_graph = []
        for key, value in graph_functions.items():
            measure_list = list()

            if key in measure_j:

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

                # handle results
                if measure['Output'] == "dict":
                    for rskey, rsval in results.items():
                        measure_list.append(rsval)
                elif measure['Output'] == "number":
                    measure_list.append(results)
                elif measure['Output'] == "tuple":
                    measure_list.append(results[0])
                elif measure['Output'] == "dict_dict":
                    for rskey, rsval in sorted(results.items()):
                        for rs2key, rs2val in sorted(rsval.items()):
                            measure_list.append(rs2val)
                elif measure['Output'] == "list":
                    measure_list.extend(results)
                elif measure['Output'] == "float_or_dict":
                    if isinstance(results, float):
                        measure_list.append(results)
                    if isinstance(results, dict):
                        for rskey, rsval in sorted(results.items()):
                            measure_list.append(rsval)
                elif measure['Output'] == "dual_tuple":
                    measure_list.append(results[0])
                    measure_list.append(results[1])
                elif measure['Output'] == "tuple_dict":
                    for rskey, rsval in sorted(results[0].items()):
                        measure_list.append(rsval)
                    for rskey, rsval in sorted(results[1].items()):
                        measure_list.append(rsval)
                if "compute_average" in measure.keys() and measure['compute_average']:
                    measure_list_graph.append(np.mean(measure_list))
                else:
                    measure_list_graph.extend(measure_list)
        return measure_list_graph

    def get_measure_info(self):
        pass

    def extract_measures(self, x_graphs, path="", ids=None):

        measure_list = []
        # check that graphs have networkx format
        if isinstance(x_graphs, np.ndarray) or isinstance(x_graphs, np.matrix):
            graphs = dense_to_networkx(x_graphs, adjacency_axis=0)
        elif isinstance(x_graphs, list):
            graphs = x_graphs
        else:
            raise TypeError("Input needs to be list of networkx graphs or numpy array.")

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        measure_json = os.path.join(base_folder, 'photonai_graph/GraphMeasures.json')
        with open(measure_json, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        if ids is not None:

            for graph, i in zip(graphs, ids):

                for key, value in self.graph_functions.items():

                    # do all the extraction steps
                    if key in measure_j:

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
                        results = getattr(networkx, key)(graph, **value)

                        # if the values are numbers:
                        # make a list of Subject ID, value, measure_name, edge=None, node=None
                        if measure['Output'] == "number":
                            list_to_append = [i, results, key, "None", "None"]
                            measure_list.append(list_to_append)
                        # if the values are dicts:
                        # for every value in the dict make a list with:
                        # subject ID, value, measure name, edge=None if it isn't an edge_method
                        if measure['Output'] == "dict":
                            for rskey, rsval in results.items():
                                if measure['node_or_edge'] == 'node':
                                    list_to_append = [i, rsval, key, rskey, "None"]
                                    measure_list.append(list_to_append)
                                elif measure['node_or_edge'] == 'edge':
                                    list_to_append = [i, rsval, key, "None", rskey]
                                    measure_list.append(list_to_append)
                        # if the values are tuples
                        elif measure['Output'] == "tuple":
                            list_to_append = [i, results[0], key, "None", "None"]
                            measure_list.append(list_to_append)
                        # if the output is a dict of dicts
                        elif measure['Output'] == "dict_dict":
                            raise NotImplementedError("Dictionary of dictionary outputs are not implemented.")
                        # if output is list: list outputs are not implemented
                        elif measure['Output'] == "list":
                            raise NotImplementedError("List outputs are not implemented")
                        # if the output is float or dict based output
                        elif measure['Output'] == "float_or_dict":
                            if isinstance(results, float):
                                list_to_append = [i, results, key, "None", "None"]
                                measure_list.append(list_to_append)
                            elif isinstance(results, dict):
                                for rskey, rsval in results.items():
                                    if measure['node_or_edge'] == "node":
                                        list_to_append = [i, rsval, key, rskey, "None"]
                                        measure_list.append(list_to_append)
                                    elif measure['node_or_edge'] == "edge":
                                        list_to_append = [i, rsval, key, "None", rskey]
                                        measure_list.append(list_to_append)
                        # if output is dual_tuple
                        elif measure['Output'] == "dual_tuple":
                            raise NotImplementedError("Dual tuple outputs are not implemented.")
                        # if output is tuple_dict
                        elif measure['Output'] == "tuple_dict":
                            raise NotImplementedError("Tuple-Dict outputs are not implemented.")

        else:
            raise Exception('no ID provided')

        df = pd.DataFrame(measure_list)

        col_names = ["ID", "value", "measure", "nodes", "edges"]

        df.to_csv(path_or_buf=path, header=col_names, index=None)
