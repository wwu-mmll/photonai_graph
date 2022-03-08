from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels
from sklearn.model_selection import KFold

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=200)
y = get_random_labels(l_type="classification", number_of_labels=200)

# Design your Pipeline
my_pipe = Hyperpipe('basic_kernel_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=5),
                    optimizer='grid_search',
                    metrics=['accuracy', 'balanced_accuracy', 'recall', 'precision'],
                    best_config_metric='accuracy')

my_pipe.add(PipelineElement('GraphConstructorThreshold', threshold=0.95))

my_pipe.add(PipelineElement('GrakelAdapter', node_feature_construction="sum"))

my_pipe.add(PipelineElement('PyramidMatch'))

my_pipe.add(PipelineElement("SVC", kernel='precomputed'))

my_pipe.fit(X, y)
