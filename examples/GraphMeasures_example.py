from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels
from sklearn.model_selection import KFold

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=10, number_of_individuals=100)
y = get_random_labels(type="regression", number_of_labels=100)

# Design your Pipeline
my_pipe = Hyperpipe('basic_gmeasure_pipe',
                    inner_cv=KFold(n_splits=3),
                    outer_cv=KFold(n_splits=3),
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')

my_pipe.add(PipelineElement('GraphConstructorThreshold',
                            hyperparameters={'threshold': 0.8}))

my_pipe.add(PipelineElement('GraphMeasureTransform',
                            hyperparameters={'graph_functions': {"large_clique_size": {},
                                                                 "global_efficiency": {},
                                                                 "overall_reciprocity": {},
                                                                 "local_efficiency": {}}}))

my_pipe.add(PipelineElement('SVR'))

my_pipe.fit(X, y)

