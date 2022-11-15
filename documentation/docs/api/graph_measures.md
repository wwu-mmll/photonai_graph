# Graph Measures

Graph measures or metrics are values that capture graph properties like efficiency. As these measures capture information across the entire graph, or for nodes, edges or subgraphs, they can be used to study graph properties. These measures can also be used as a low-dimensional representation of the graph in machine learning tasks. Since graph measures capture information across the entire graph, they make use of information that would otherwise be lost if one chose to only use node features or a vectorized connectivity matrix for example.


## GraphMeasureTransform
::: photonai_graph.Measures.NetworkxMeasureTransform.NetworkxMeasureTransform.__init__
    rendering:
        show_root_toc_entry: False

## Available Measures

The following measures are available. They are different networkx functions that calculate these measures. Be warned that, depending on your graph, these measures might take very long to calculate. Also some measures are only available as part of a pipeline, while some are also available for extraction (see accompanying table).

| Measure | Name in Dict | available for extract |
| ----- | ----- | ----- |
| Node connectivity for all pairs | all_pairs_node_connectivity | Yes |
| Local node connectivity | local_node_connectivity | Yes |
| Node connectivity | node_connectivity | Yes |
| K component structure | k_components | No |
| large clique size | large_clique_size | Yes |
| Average clustering Coefficient | average_clustering | Yes |
| Treewidth decomposition (Minimum Degree Heuristic) | treewidth_min_degree | No |
| Treewidth decomposition (Minimum Fill-in heuristic) | treewidth_min_fill_in | No |
| Degree assortativity of graph | degree_assortativity_coefficient | Yes |
| Attribute assortativity of graph | attribute_assortativity_coefficient | Yes |
| Numeric assortativity of graph | numeric_assortativity_coefficient | Yes |
| Degree assortaivity of graph (scipy implmenetation) | degree_pearson_correlation_coefficient | Yes |
| Average neighbour degree node-wise | average_neighbor_degree | Yes |
| Average degree connectivity of graph | average_degree_connectivity | No |
| Average degree connectivity of graph (kNN implementation) | k_nearest_neighbors | No |
| Degree centrality (node-wise) | degree_centrality | Yes |
| In-degree centrality (node-wise) | in_degree_centrality | Yes |
| Out-degree centrality (node-wise) | out_degree_centrality | Yes |
| Eigenvector centrality (node-wise) | eigenvector_centrality | Yes |
| Katz centrality (node-wise) | katz_centrality | Yes |
| Closeness centrality (node-wise) | closeness_centrality | Yes |
| Incremental closeness centrality (node-wise) | incremental_closeness_centrality | Yes |
| Current flow closeness centrality (node-wise) | current_flow_closeness_centrality | Yes |
| Information centrality (node-wise) | information_centrality | Yes |
| Betweeness centrality (node-wise) | betweenness_centrality | Yes |
| Betweeness centrality (node-wise) on a subset of nodes | betweenness_centrality_subset | No |
| Betweeness centrality (edge-wise) on a subset of edges | edge_betweenness_centrality_subset | No |
| Current flow betweeness centrality (node-wise) | current_flow_betweenness_centrality | Yes |
| Current flow betweeness centrality (edge-wise) | enedge_current_flow_betweenness_centrality | Yes |
| Approximate current flow betweeness centrality (node-wise) | approximate_current_flow_betweenness_centrality | Yes | 
| Current flow betweeness centrality (node-wise) on a subset of nodes | current_flow_betweenness_centrality_subset | Yes |
| Current flow betweeness centrality (edge-wise) on a subset of edges | edge_current_flow_betweenness_centrality_subset | Yes |
| Communicability betweeness centrality (noe-wise) | communicability_betweenness_centrality | Yes |
| Betweeness centrality (group-wise) | group_betweenness_centrality | Yes |
| Closeness centrality (group-wise) | group_closeness_centrality | Yes |
| Degree centrality (group-wise) | group_degree_centrality | Yes |
| In-degree centrality (group-wise) | group_in_degree_centrality | Yes |
| Out-degree centrality (group-wise) | group_out_degree_centrality | Yes |
| Load centrality (node-wise) | load_centrality | Yes |
| Load centrality (edge-wise) | edge_load_centrality | Yes |
| Subgraph centrality (node-wise) | subgraph_centrality | Yes |
| Subgraph centrality (node-wise, exponent implementation) | subgraph_centrality_exp | Yes |
| Estrada Index | estrada_index | Yes |
| Harmonic centrality (node-wise) | harmonic_centrality | Yes |
| Dispersion | dispersion | No |
| Local reaching centrality | local_reaching_centrality | Yes |
| Global reaching centrality | global_reaching_centrality | Yes |
| Perlocation centrality (node-wise) | percolation_centrality | Yes |
| Second-order centrality (node-wise) | second_order_centrality | Yes |
| Voterank via VoteRank algorithm | voterank | No |
| Size of largest clique | graph_clique_number | Yes |
| Number of maximal cliques in graph | graph_number_of_cliques | Yes |
| Size of largest clique containing each specified node | node_clique_number | Yes |
| Number of cliques for each specified node | number_of_cliques | Yes |
| Number of Triangles | triangles | Yes |
| Transivity | transitivity | Yes |
| Clustering coefficient for specified nodes | clustering | No |
| Square Clustering (node-wise) | square_clustering | Yes |
| Communicability | communicability | No |
| Communicability (exponent implmenetation) | communicability_exp | No |
| Number of connected components | number_connected_components | Yes |
| Number of strongly connected components | number_strongly_connected_components | Yes |
| Number of weakly connected components | number_weakly_connected_components | Yes |
| Number of attracting components | number_attracting_components | Yes |
| Average Node connectivity | average_node_connectivity | Yes |
| Edge connectivity | edge_connectivity | Yes |
| Local Edge connectivity | local_edge_connectivity | Yes |
| Core number (node-wise) | core_number | Yes |
| Onion layers (node-wise) | onion_layers | Yes |
| Boundary expansion (set-wise) | boundary_expansion | Yes |
| Conductance between two sets of nodes | conductance | Yes |
| Cut size between two sets of nodes | cut_size | Yes |
| Edge expansion between two sets of nodes | edge_expansion | Yes |
| Mixing expansion between two sets of nodes | mixing_expansion | Yes |
| Volume between two sets of nodes | volume | Yes |
| Diameter | diameter | Yes |
| Eccentricity | eccentricity | Yes |
| Radius | radius | Yes |
| Resistance Distance between node A and B | resistance_distance | Yes |
| Efficiency between node A and B | efficiency | Yes |
| Local efficiency | local_efficiency | Yes |
| Global efficiency | global_efficiency | Yes |
| Flow hierarchy | flow_hierarchy | Yes |
| Number of isolates | number_of_isolates | Yes |
| PageRank value (node-wise) | pagerank | Yes |
| HITS hubs and authorities | hits | No |
| Non-randomness | non_randomness | No |
| Reciprocity (node-wise) | reciprocity | Yes |
| Overall reciprocity | overall_reciprocity | Yes |
| Rich-club coefficient | rich_club_coefficient | Yes |
| Average shortest path length | average_shortest_path_length | Yes |
| Small-world coefficient sigma | sigma | Yes |
| Small-world coefficient omega | omega | Yes |
| S-metric | s_metric | Yes |
| Constraint (node-wise) | constraint | Yes |
| Effective Size (node-wise) | effective_size | Yes |
| Local constraint of node A with respect to node B | local_constraint | Yes |
| Triadic census | triadic_census | No |
| Closeness vitality | closeness_vitality | No |
| Wiener Index | wiener_index | Yes |
