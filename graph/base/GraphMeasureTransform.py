"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for extracting graph measures that can then be
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

#TODO: make error messages for possible errors
#TODO: make the import section a bit more "airy"
#TODO: make documentation for every single method

import networkx
from networkx.algorithms import approximation
from networkx.algorithms.assortativity import degree_assortativity_coefficient, average_degree_connectivity, attribute_assortativity_coefficient
from networkx.algorithms.assortativity import numeric_assortativity_coefficient, degree_pearson_correlation_coefficient, average_neighbor_degree
from networkx.algorithms.assortativity import k_nearest_neighbors
from networkx.algorithms.centrality import degree_centrality, eigenvector_centrality, katz_centrality, closeness_centrality, current_flow_closeness_centrality
from networkx.algorithms.centrality import information_centrality, in_degree_centrality, out_degree_centrality, incremental_closeness_centrality
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality, load_centrality
from networkx.algorithms.centrality import edge_load_centrality, betweenness_centrality_subset, edge_betweenness_centrality_subset, current_flow_betweenness_centrality
from networkx.algorithms.centrality import edge_current_flow_betweenness_centrality, approximate_current_flow_betweenness_centrality, current_flow_betweenness_centrality_subset
from networkx.algorithms.centrality import edge_current_flow_betweenness_centrality_subset, communicability_betweenness_centrality, group_betweenness_centrality
from networkx.algorithms.centrality import group_closeness_centrality, group_degree_centrality, \
    group_out_degree_centrality
from networkx.algorithms.centrality import subgraph_centrality, subgraph_centrality_exp, estrada_index, harmonic_centrality, dispersion, local_reaching_centrality
from networkx.algorithms.centrality import global_reaching_centrality, percolation_centrality, second_order_centrality, voterank
from networkx.algorithms import clique, cluster, communicability_alg, components, connectivity, core, cuts, distance_measures, dominance, efficiency_measures, hierarchy
from networkx.algorithms import isolate, link_analysis, non_randomness, reciprocity, richclub, shortest_paths, smallworld, smetric, structuralholes, triads, vitality, wiener
from sklearn.base import BaseEstimator, TransformerMixin
from photonai.graph.base.GraphUtilities import DenseToNetworkx
import pandas as pd
import numpy as np
import json
import os


class GraphMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 graph_functions = {"global_efficiency": {},
                                    "sigma": {}},
                 logs=''):

        self.graph_functions = graph_functions

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        X_transformed = []

        graphs = DenseToNetworkx(X, adjacency_axis=0)

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        measureJSON = os.path.join(base_folder, 'photonai/graph/base/GraphMeasures.json')
        with open(measureJSON, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        for i in graphs:

            measure_list = []

            for key, value in self.graph_functions.items():

                if key in measure_j:

                    measure = measure_j[key]
                    # remove self loops if not allowed
                    if measure['self_loops_allowed'] == False:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                    # make graph directed or undirected depending on what is needed
                    if measure['Undirected'] == True:
                        i.to_undirected()
                    elif measure['Undirected'] == False:
                        i.to_directed()
                     # call function
                    results = getattr(networkx, key)(i, **value)

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
                        if hasattr(results, float):
                            measure_list.append(results)
                        if hasattr(results, dict):
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


            X_transformed.append(measure_list)

        np.asarray(X_transformed)

        return X_transformed


    def get_measure_info(self):
        pass

    def extract_measures(self, X, path="", IDs=None):

        measure_list = []
        # turn graphs into networkx format
        graphs = DenseToNetworkx(X, adjacency_axis=0)

        # load json file
        base_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        measureJSON = os.path.join(base_folder, 'photonai/graph/base/GraphMeasures.json')
        with open(measureJSON, 'r') as measure_json_file:
            measure_j = json.load(measure_json_file)

        if not IDs == None:

            for graph, i in zip(graphs, IDs):

                for key, value in self.graph_functions.items():

                    # do all the extraction steps
                    if key in measure_j:

                        measure = measure_j[key]
                        # remove self loops if not allowed
                        if measure['self_loops_allowed'] == False:
                            graph.remove_edges_from(networkx.selfloop_edges(graph))
                        # make graph directed or undirected depending on what is needed
                        if measure['Undirected'] == True:
                            graph.to_undirected()
                        elif measure['Undirected'] == False:
                            graph.to_directed()
                        # call function
                        results = getattr(networkx, key)(graph, **value)

                        # if the values are numbers: make a list of Subject ID, value, measure_name, edge=None, node=None
                        if measure['Output'] == "number":
                            list_to_append = [i, results, key, "None", "None"]
                            measure_list.append(list_to_append)
                        # if the values are dicts: for every value in the dict make a list with subject ID, value, measure name, edge=None if it isn't an edge_method
                        if measure['Output'] == "dict":
                            for rskey, rsval in results.items():
                                if measure['node_or_edge'] == 'node':
                                    list_to_append = [i, rsval, key, rskey, "None", "None"]
                                    measure_list.append(list_to_append)
                                elif measure['node_or_edge'] == 'edge':
                                    list_to_append = [i, rsval, key, "None", rskey, "None"]
                                    measure_list.append(list_to_append)
                        # if the values are tuples
                        elif measure['Output'] == "tuple":
                            list_to_append = [i, results[0], key, "None", "None", "None"]
                            measure_list.append(list_to_append)
                        # if the output is a dict of dicts
                        elif measure['Output'] == "dict_dict":
                            raise Exception("Dictionary of dictionary outputs are not implemented.")
                        # if output is list: list outputs are not implemented
                        elif measure['Output'] == "list":
                            raise Exception("List outputs are not implemented")
                        # if the output is float or dict based output
                        elif measure['Output'] == "float_or_dict":
                            if hasattr(results, float):
                                list_to_append = [i, results, key, "None", "None"]
                                measure_list.append(list_to_append)
                            if hasattr(results, dict):
                                if measure['node_or_edge'] == 'node':
                                    list_to_append = [i, rsval, key, rskey, "None", "None"]
                                    measure_list.append(list_to_append)
                                if measure['node_or_edge'] == 'edge':
                                    list_to_append = [i, rsval, key, "None", rskey, "None"]
                                    measure_list.append(list_to_append)
                        # if output is dual_tuple
                        elif measure['Output'] == "dual_tuple":
                            raise Exception("Dual tuple outputs are not implemented.")
                        # if output is tuple_dict
                        elif measure['Output'] == "tuple_dict":
                            raise Exception("Tuple-Dict outputs are not implemented.")

        else:
            raise Exception('no ID provided')

        df = pd.DataFrame(measure_list)

        df.to_csv(path_or_buf=path)

