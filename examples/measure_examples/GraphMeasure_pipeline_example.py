from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels
from sklearn.model_selection import KFold
from photonai_graph.GraphMeasureTransform import GraphMeasureTransform

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=200)
y = get_random_labels(l_type="regression", number_of_labels=200)

# Design your Pipeline
my_pipe = Hyperpipe('basic_measure_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=5),
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')

my_pipe.add(PipelineElement('GraphConstructorThreshold',
                            hyperparameters={'threshold': 0.95}))

my_pipe.add(PipelineElement('GraphMeasureTransform',
                            hyperparameters={'graph_functions': {"global_efficiency": {},
                                                                 "local_efficiency": {},
                                                                 "average_clustering": {},
                                                                 "degree_centrality": {},
                                                                 "betweenness_centrality": {},
                                                                 "overall_reciprocity": {}}}))

my_pipe.add(PipelineElement("SVR"))

my_pipe.fit(X, y)