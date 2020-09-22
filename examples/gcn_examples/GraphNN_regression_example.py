from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels
from sklearn.model_selection import KFold

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=100)
y = get_random_labels(l_type="regression", number_of_labels=100)

# Design your Pipeline
my_pipe = Hyperpipe('basic_gembedding_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=5),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')

my_pipe.add(PipelineElement('GraphConstructorPercentage',
                            hyperparameters={'percentage': 90.0}))

my_pipe.add(PipelineElement('GATRegressor',
                            hyperparameters={'nn_epochs': 20}))

my_pipe.fit(X, y)