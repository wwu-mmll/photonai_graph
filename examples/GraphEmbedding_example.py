from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, RegisterGraph_force
from sklearn.model_selection import KFold
import numpy as np

# register photon photonai_graph with the photonai_graph force function
RegisterGraph_force()

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=400)
y = np.random.rand(10)

# Design your Pipeline
my_pipe = Hyperpipe('basic_svm_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error')

my_pipe.add(PipelineElement('GraphConstructorThreshold',
                            hyperparameters={'threshold': 0.95}))

my_pipe.add(PipelineElement('GraphEmbeddingHOPE'))

my_pipe.add(PipelineElement('SVR'))

my_pipe.fit(X, y)