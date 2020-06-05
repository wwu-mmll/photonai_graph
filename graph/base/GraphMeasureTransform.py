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
import numpy as np
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

        graphs = DenseToNetworkx(X, adjacency_axis = 0)

        # loop over each individual in the graph
        for i in graphs:

            # initialize list of measures you are about to compute
            measure_list = []

            # iterate over dict and check which keywords are there
            for key, value in self.graph_functions.items():
                # if there is keyword match

                # approximation all_pairs_node_connectivity
                if key == "all_pairs_node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        all_pairs_node_connectivity_results = approximation.all_pairs_node_connectivity(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in all_pairs_node_connectivity_results.items():
                        measure_list.append(rsval)

                # approximation local_node_connectivity
                if key == "local_node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        local_node_connectivity_results = approximation.local_node_connectivity(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in local_node_connectivity_results.items():
                        measure_list.append(rsval)

                # approximation node_connectivity
                if key == "node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        node_connectivity_results = approximation.node_connectivity(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in node_connectivity_results.items():
                        measure_list.append(rsval)

                #approximation k_components
                if key == "k_components":
                    # run function with specified parameters
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        k_components_results = approximation.k_components(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    # handle the results
                    for rskey, rsval in k_components_results.items():
                        # append to the graph measure list
                        measure_list.append(rsval)

                #approximation large_clique_size
                if key == "large_clique_size":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        #large clique size function does not take any arguments and returns an integer
                        large_clique_size_result = approximation.large_clique_size(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(large_clique_size_result)

                #approximation average_clustering
                if key == "average_clustering":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        #average clustering returns a float
                        average_clustering_results = approximation.average_clustering(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(average_clustering_results)

                # approximation treewidth_min_degree
                if key == "treewidth_min_degree":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        treewidth_min_degree_results = approximation.treewidth_min_degree(i)
                    except TypeError:
                        print(
                            'Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(treewidth_min_degree_results[0])

                # approximaiton treewidth_min_fill_in
                if key == "treewidth_min_fill_in":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        treewidth_min_fill_in_results = approximation.treewidth_min_fill_in(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(treewidth_min_fill_in_results[0])

                # assortativity degree_assortativity_coefficient
                if key == "degree_assortativity_coefficient":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        if any (k in value for k in ("x","y")):
                            i.to_directed()
                        degree_assortativity_coefficient_results = degree_assortativity_coefficient(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(degree_assortativity_coefficient_results)

                # assortativity attribute_assortativity_coefficient
                if key == "attribute_assortativity_coefficient":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        attribute_assortativity_coefficient_results = attribute_assortativity_coefficient(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(attribute_assortativity_coefficient_results)

                # assortativity numeric_assortativity_coefficient
                if key == "numeric_assortativity_coefficient":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        x_results = numeric_assortativity_coefficient(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in x_results.items():
                        measure_list.append(rsval)

                # assortativity degree_pearson_correlation_coefficient
                if key == "degree_pearson_correlation_coefficient":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        if any (k in value for k in ("x","y")):
                            i.to_directed()
                        degree_pearson_correlation_coefficient_results = degree_pearson_correlation_coefficient(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(degree_pearson_correlation_coefficient_results)

                # assortativity average_neighbor_degree
                if key == "average_neighbor_degree":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        if any (k in value for k in ("source","target")):
                            i.to_directed()
                        average_neighbor_degree_results = average_neighbor_degree(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in average_neighbor_degree_results.items():
                        measure_list.append(rsval)

                # assortativity average_degree_connectivity
                if key == "average_degree_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        if any(k in value for k in ("source", "target")):
                            i.to_directed()
                        average_degree_connectivity_results = average_degree_connectivity(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in average_degree_connectivity_results.items():
                        measure_list.append(rsval)

                # assortativity k_nearest_neighbors
                if key == "k_nearest_neighbors":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        if any(k in value for k in ("source", "target")):
                            i.to_directed()
                        k_nearest_neighbors_results = k_nearest_neighbors(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in k_nearest_neighbors_results.items():
                        measure_list.append(rsval)

                # centrality degree_centrality
                if key == "degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        degree_centrality_results = degree_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in degree_centrality_results.items():
                        measure_list.append(rsval)

                # centrality in_degree_centrality
                if key == "in_degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        in_degree_centrality_results = in_degree_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in in_degree_centrality_results.items():
                        measure_list.append(rsval)

                # centrality out_degree_centrality
                if key == "out_degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        out_degree_centrality_results = out_degree_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in out_degree_centrality_results.items():
                        measure_list.append(rsval)

                # centrality eigenvector_centrality
                if key == "eigenvector_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        eigenvector_centrality_results = eigenvector_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(eigenvector_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality katz_centrality
                if key == "katz_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        katz_centrality_results = katz_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(katz_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality closeness_centrality
                if key == "closeness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        closeness_centrality_results = closeness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(closeness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality incremental_closeness_centrality
                if key == "incremental_closeness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        incremental_closeness_centrality_results = incremental_closeness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(incremental_closeness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality current_flow_closeness_centrality
                if key == "current_flow_closeness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        current_flow_closeness_centrality_results = current_flow_closeness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(current_flow_closeness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality information_centrality
                if key == "information_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        information_centrality_results = information_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(information_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality betweenness_centrality
                if key == "betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        betweenness_centrality_results = betweenness_centrality(i, **value)
                    except TypeError:
                        print(
                            'Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality edge_betweenness_centrality
                if key == "edge_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_betweenness_centrality_results = edge_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality betweenness_centrality_subset
                if key == "betweenness_centrality_subset":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        betweenness_centrality_subset_results = betweenness_centrality_subset(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality edge_betweenness_centrality_subset
                if key == "edge_betweenness_centrality_subset":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_betweenness_centrality_subset_results = edge_betweenness_centrality_subset(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(edge_betweenness_centrality_subset_results.items()):
                        measure_list.append(rsval)

                # centrality current_flow_betweenness_centrality
                if key == "current_flow_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        current_flow_betweenness_centrality_results = current_flow_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(current_flow_betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality edge_current_flow_betweenness_centrality
                if key == "edge_current_flow_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_current_flow_betweenness_centrality_results = edge_current_flow_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(edge_current_flow_betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality approximate_current_flow_betweenness_centrality
                if key == "approximate_current_flow_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        approximate_current_flow_betweenness_centrality_results = approximate_current_flow_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(approximate_current_flow_betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality current_flow_betweenness_centrality_subset
                if key == "current_flow_betweenness_centrality_subset":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        current_flow_betweenness_centrality_subset_results = current_flow_betweenness_centrality_subset(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(current_flow_betweenness_centrality_subset_results.items()):
                        measure_list.append(rsval)

                # centrality edge_current_flow_betweenness_centrality_subset
                if key == "edge_current_flow_betweenness_centrality_subset":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_current_flow_betweenness_centrality_subset_results = edge_current_flow_betweenness_centrality_subset(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(edge_current_flow_betweenness_centrality_subset_results.items()):
                        measure_list.append(rsval)

                # centrality communicability_betweenness_centrality
                if key == "communicability_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        communicability_betweenness_centrality_results = communicability_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(communicability_betweenness_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality group_betweenness_centrality
                if key == "group_betweenness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        group_betweenness_centrality_results = group_betweenness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(group_betweenness_centrality_results)

                # centrality group_closeness_centrality
                if key == "group_closeness_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        group_closeness_centrality_results = group_closeness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(group_closeness_centrality_results)

                # centrality group_degree_centrality
                if key == "group_degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        group_degree_centrality_results = group_degree_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(group_degree_centrality_results)

                # centrality group_in_degree_centrality
                if key == "group_in_degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        group_closeness_centrality_results = group_closeness_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(group_closeness_centrality_results)

                # centrality group_out_degree_centrality
                if key == "group_out_degree_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        group_out_degree_centrality_results = group_out_degree_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(group_out_degree_centrality_results)

                # centrality load_centrality
                if key == "load_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        load_centrality_results = load_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in load_centrality_results.items():
                        measure_list.append(rsval)

                # centrality edge_load_centrality
                if key == "edge_load_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_load_centrality_results = edge_load_centrality(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in edge_load_centrality_results.items():
                        measure_list.append(rsval)

                # centrality subgraph_centrality
                if key == "subgraph_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        subgraph_centrality_results = subgraph_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(subgraph_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality subgraph_centrality_exp
                if key == "subgraph_centrality_exp":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        subgraph_centrality_exp_results = subgraph_centrality_exp(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(subgraph_centrality_exp_results.items()):
                        measure_list.append(rsval)

                # centrality estrada_index
                if key == "estrada_index":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        estrada_index_results = estrada_index(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(estrada_index_results)

                # centrality harmonic_centrality
                if key == "harmonic_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        harmonic_centrality_results = harmonic_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(harmonic_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality dispersion
                if key == "dispersion":
                    try:
                        if any (k in value for k in ("u", "v")):
                            i.remove_edges_from(networkx.selfloop_edges(i))
                            dispersion_results = dispersion(i, **value)
                            for rskey, rsval in sorted(dispersion_results.items()):
                                measure_list.append(rsval)
                        else:
                            i.remove_edges_from(networkx.selfloop_edges(i))
                            dispersion_results = dispersion(i, **value)
                            for rskey, rsval in sorted(dispersion_results.items()):
                                for rs2key, rs2val in sorted(rsval.items()):
                                    measure_list.append(rs2val)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # centrality local_reaching_centrality
                if key == "local_reaching_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        local_reaching_centrality_results = local_reaching_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(local_reaching_centrality_results)

                # centrality global_reaching_centrality
                if key == "global_reaching_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        i.to_directed()
                        global_reaching_centrality_results = global_reaching_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(global_reaching_centrality_results)

                # centrality percolation_centrality
                if key == "percolation_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        percolation_centrality_results = percolation_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(global_reaching_centrality_results)

                # centrality second_order_centrality
                if key == "second_order_centrality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        second_order_centrality_results = second_order_centrality(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(second_order_centrality_results.items()):
                        measure_list.append(rsval)

                # centrality voterank
                if key == "voterank":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        voterank_results = voterank(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.extend(voterank_results)

                # cliques graph_clique_number
                if key == "graph_clique_number":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        # we are using all cliques as we are not passing any parameters to the graph_clique_number function
                        graph_clique_number_results = clique.graph_clique_number(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(graph_clique_number_results)

                # cliques graph_number_of_cliques
                if key == "graph_number_of_cliques":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        # we are using all cliques as we are not passing any parameters to the graph_number_of_cliques function
                        graph_number_of_cliques_results = clique.graph_number_of_cliques(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.append(graph_number_of_cliques_results)

                # cliques node_clique_number
                if key == "node_clique_number":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        # we are using all cliques as we are not passing any cliques to the graph_number_of_cliques function
                        node_clique_number_results = clique.node_clique_number(i)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.extend(node_clique_number_results)

                # cliques number_of_cliques
                if key == "number_of_cliques":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        # we are passing arguments to to the number_of_cliques function, but here we can only use the nodes (not the clique function)
                        number_of_cliques_results = clique.number_of_cliques(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    measure_list.extend(number_of_cliques_results)

                # clustering triangles
                if key == "triangles":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        # we are passing arguments to to the number_of_cliques function, but here we can only use the nodes (not the clique function)
                        triangles_results = cluster.triangles(i, **value)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')
                    for rskey, rsval in sorted(triangles_results.items()):
                        measure_list.append(rsval)

                # clustering transitivity
                if key == "transitivity":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    transitivity_results = cluster.transitivity(i)
                    measure_list.append(transitivity_results)

                # clustering clustering
                if key == "clustering":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        clustering_results = cluster.clustering(i, **value)
                        if hasattr(clustering_results, float):
                            measure_list.append(clustering_results)
                        if hasattr(clustering_results, dict):
                            for rskey, rsval in sorted(clustering_results.items()):
                                measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # clustering average_clustering
                if key == "average_clustering":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        average_clustering_results = cluster.average_clustering(i, **value)
                        measure_list.append(average_clustering_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')


                # clustering square_clustering
                if key == "square_clustering":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        square_clustering_results = cluster.square_clustering(i, **value)
                        for rskey, rsval in sorted(square_clustering_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # communicability communicability
                if key == "communicability":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        communicability_results = communicability_alg.communicability(i, **value)
                        for rskey, rsval in sorted(communicability_results.items()):
                            for rs2key, rs2val in sorted(rsval.items()):
                                measure_list.append(rs2val)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # communicability communicability_exp
                if key == "communicability_exp":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        communicability_exp_results = communicability_alg.communicability_exp(i, **value)
                        for rskey, rsval in sorted(communicability_exp_results.items()):
                            for rs2key, rs2val in sorted(rsval.items()):
                                measure_list.append(rs2val)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # components number_connected_components
                if key == "number_connected_components":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    number_connected_components_results = components.number_connected_components(i)
                    measure_list.append(number_connected_components_results)

                # components number_strongly_connected_components
                if key == "number_strongly_connected_components":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    i.to_directed()
                    number_strongly_connected_components_results = components.number_strongly_connected_components(i)
                    measure_list.append(number_strongly_connected_components_results)

                # components number_weakly_connected_components
                if key == "number_weakly_connected_components":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    i.to_directed()
                    number_weakly_connected_components_results = components.number_weakly_connected_components(i)
                    measure_list.append(number_weakly_connected_components_results)

                # components number_attracting_components
                if key == "number_attracting_components":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    i.to_directed()
                    number_attracting_components_results = components.number_attracting_components(i)
                    measure_list.append(number_attracting_components_results)

                # connectivity average_node_connectivity
                if key == "average_node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        average_node_connectivity_results = connectivity.average_node_connectivity(i, **value)
                        measure_list.append(average_node_connectivity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # connectivity all_pairs_node_connectivity
                if key == "all_pairs_node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        all_pairs_node_connectivity_results = connectivity.all_pairs_node_connectivity(i, **value)
                        for rskey, rsval in sorted(all_pairs_node_connectivity_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # connectivity edge_connectivity
                if key == "edge_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        edge_connectivity_results = connectivity.edge_connectivity(i, **value)
                        measure_list.append(edge_connectivity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # connectivity local_edge_connectivity
                if key == "local_edge_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        local_edge_connectivity_results = connectivity.local_edge_connectivity(i, **value)
                        measure_list.append(local_edge_connectivity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # connectivity local_node_connectivity
                if key == "local_node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        local_node_connectivity_results = connectivity.local_node_connectivity(i, **value)
                        measure_list.append(local_node_connectivity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # connectivity node_connectivity
                if key == "node_connectivity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        node_connectivity_results = connectivity.node_connectivity(i, **value)
                        measure_list.append(node_connectivity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')


                # core core_number
                if key == "all_pairs_node_connectivity":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    core_number_results = core.core_number(i)
                    for rskey, rsval in sorted(core_number_results.items()):
                        measure_list.append(rsval)

                # core onion_layers
                if key == "onion_layers":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    onion_layers_results = core.onion_layers(i)
                    for rskey, rsval in sorted(onion_layers_results.items()):
                        measure_list.append(rsval)

                # cuts boundary_expansion
                if key == "boundary_expansion":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    boundary_expansion_results = cuts.boundary_expansion(i, **value)
                    measure_list.append(boundary_expansion_results)

                # cuts conductance
                if key == "conductance":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    conductance_results = cuts.conductance(i, **value)
                    measure_list.append(conductance_results)

                # cuts cut_size
                if key == "cut_size":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    cut_size_results = cuts.cut_size(i, **value)
                    measure_list.append(cut_size_results)

                # cuts edge_expansion
                if key == "edge_expansion":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    edge_expansion_results = cuts.edge_expansion(i, **value)
                    measure_list.append(edge_expansion_results)

                # cuts mixing_expansion
                if key == "mixing_expansion":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    mixing_expansion_results = cuts.mixing_expansion(i, **value)
                    measure_list.append(mixing_expansion_results)

                # cuts node_expansion
                if key == "node_expansion":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    node_expansion_results = cuts.node_expansion(i, **value)
                    measure_list.append(node_expansion_results)

                # cuts normalized_cut_size
                if key == "normalized_cut_size":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    normalized_cut_size_results = cuts.normalized_cut_size(i, **value)
                    measure_list.append(normalized_cut_size_results)

                # cuts volume
                if key == "volume":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    volume_results = cuts.volume(i, **value)
                    measure_list.append(volume_results)

                # distance_measures diameter
                if key == "diameter":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        diameter_results = distance_measures.diameter(i, **value)
                        measure_list.append(diameter_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # distance_measures eccentricity
                if key == "eccentricity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        eccentricity_results = distance_measures.eccentricity(i, **value)
                        measure_list.append(eccentricity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # distance_measures extrema_bounding
                if key == "extrema_bounding":
                    try:
                        if 'center' in value.values() or 'periphery' in value.values():
                            i.remove_edges_from(networkx.selfloop_edges(i))
                            extrema_bounding_results = distance_measures.extrema_bounding(i, **value)
                            for rskey, rsval in sorted(extrema_bounding_results.items()):
                                measure_list.append(rsval)
                        else:
                            i.remove_edges_from(networkx.selfloop_edges(i))
                            extrema_bounding_results = distance_measures.extrema_bounding(i, **value)
                            measure_list.append(extrema_bounding_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # distance measures radius
                if key == "radius":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    radius_results = distance_measures.radius(i, **value)
                    measure_list.append(radius_results)

                # distance_measures resistance_distance
                if key == "resistance_distance":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        resistance_distance_results = distance_measures.resistance_distance(i, **value)
                        measure_list.append(resistance_distance_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # dominance immediate_dominators
                if key == "immediate_dominators":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        immediate_dominators_results = dominance.immediate_dominators(i, **value)
                        for rskey, rsval in sorted(immediate_dominators_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # dominance dominance_frontiers
                if key == "dominance_frontiers":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        dominance_frontiers_results = dominance.dominance_frontiers(i, **value)
                        for rskey, rsval in sorted(dominance_frontiers_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # efficiency_measures efficiency
                if key == "efficiency":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        efficiency_results = efficiency_measures.efficiency(i, **value)
                        measure_list.append(efficiency_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # efficiency_measures local_efficiency
                if key == "local_efficiency":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        local_efficiency_results = efficiency_measures.local_efficiency(i, **value)
                        measure_list.append(local_efficiency_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # efficiency_measures global_efficiency
                if key == "global_efficiency":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        global_efficiency_results = efficiency_measures.global_efficiency(i, **value)
                        measure_list.append(global_efficiency_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # hierarchy flow_hierarchy
                if key == "flow_hierarchy":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        flow_hierarchy_results = hierarchy.flow_hierarchy(i, **value)
                        measure_list.append(flow_hierarchy_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # isolate number_of_isolates
                if key == "number_of_isolates":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    number_of_isolates_results = isolate.number_of_isolates(i)
                    measure_list.append(number_of_isolates_results)

                # link_analysis pagerank
                if key == "pagerank":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        pagerank_results = link_analysis.pagerank(i, **value)
                        for rskey, rsval in sorted(pagerank_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # link_analysis hits
                if key == "hits":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        hits_results = link_analysis.hits(i, **value)
                        hits_results_hubs = hits_results[0]
                        hits_results_authorities = hits_results[1]
                        for rskey, rsval in sorted(hits_results_hubs.items()):
                            measure_list.append(rsval)
                        for rskey, rsval in sorted(hits_results_authorities.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # non_randomness non_randomness
                if key == "non_randomness":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        non_randomness_results = non_randomness.non_randomness(i, **value)
                        measure_list.append(non_randomness_results[0])
                        measure_list.append(non_randomness_results[1])
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # reciprocity overall_reciprocity
                if key == "overall_reciprocity":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    overall_reciprocity_results = reciprocity.overall_reciprocity(i)
                    measure_list.append(overall_reciprocity_results)

                # reciprocity reciprocity
                if key == "reciprocity":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        reciprocity_results = reciprocity.reciprocity(i, **value)
                        measure_list.append(reciprocity_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')


                # richclub rich_club_coefficient
                if key == "rich_club_coefficient":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        rich_club_coefficient_results = richclub.rich_club_coefficient(i, **value)
                        measure_list.append(rich_club_coefficient_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                 # shortest_path average_shortest_path_length
                if key == "average_shortest_path_length":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        average_shortest_path_length_results = shortest_paths.average_shortest_path_length(i, **value)
                        measure_list.append(average_shortest_path_length_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # smallworld omega
                if key == "omega":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        omega_results = smallworld.omega(i, **value)
                        measure_list.append(omega_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # smallworld sigma
                if key == "sigma":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        sigma_results = smallworld.sigma(i, **value)
                        measure_list.append(sigma_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # smetric s_metric
                if key == "s_metric":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        s_metric_results = smetric.s_metric(i, **value)
                        measure_list.append(s_metric_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # structuralholes constraint
                if key == "constraint":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        constraint_results = structuralholes.constraint(i, **value)
                        for rskey, rsval in sorted(constraint_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # structuralholes effective_size
                if key == "effective_size":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        effective_size_results = structuralholes.effective_size(i, **value)
                        for rskey, rsval in sorted(effective_size_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # structuralholes local_constraint
                if key == "local_constraint":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        local_constraint_results = structuralholes.local_constraint(i, **value)
                        measure_list.append(local_constraint_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # triads triadic_census
                if key == "triadic_census":
                    i.remove_edges_from(networkx.selfloop_edges(i))
                    triadic_census_results = triads.triadic_census(i)
                    for rskey, rsval in sorted(triadic_census_results.items()):
                        measure_list.append(rsval)

                # vitality closeness_vitality
                if key == "closeness_vitality":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        closeness_vitality_results = vitality.closeness_vitality(i, **value)
                        for rskey, rsval in sorted(triadic_census_results.items()):
                            measure_list.append(rsval)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')

                # wiener wiener_index
                if key == "wiener_index":
                    try:
                        i.remove_edges_from(networkx.selfloop_edges(i))
                        wiener_index_results = wiener.wiener_index(i, **value)
                        measure_list.append(wiener_index_results)
                    except TypeError:
                        print('Initializing function with your parameters crashed. Check if your argument names are correct.')


            X_transformed.append(measure_list)

        np.asarray(X_transformed)

        return X_transformed

