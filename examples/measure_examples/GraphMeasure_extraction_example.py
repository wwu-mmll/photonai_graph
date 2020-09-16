from photonai_graph.GraphUtilities import get_random_connectivity_data
from photonai_graph.GraphMeasureTransform import GraphMeasureTransform
from photonai_graph.GraphConstruction.graph_constructor_threshold import GraphConstructorThreshold

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=200)


# instantiate a constructor to threshold the graphs
g_constructor = GraphConstructorThreshold(threshold=0.95, transform_style="individual")

# instantiate the measure_transformer
g_measure_transformer = GraphMeasureTransform(graph_functions={"global_efficiency": {},
                                                               "local_efficiency": {},
                                                               "average_clustering": {},
                                                               "degree_centrality": {},
                                                               "betweenness_centrality": {},
                                                               "overall_reciprocity": {}})

X_trans = g_constructor.transform(X)

g_measure_transformer.extract_measures(X_trans, path="/path/to/your/data/test_measures.csv")
