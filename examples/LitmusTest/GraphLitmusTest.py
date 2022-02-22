from Networkx_Dataset import get_random_graphs, plot_nx_edge_count
from photonai.base import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold
from photonai_graph.GraphConversions import networkx_to_dense
import numpy as np

# create dataset
sparse_graphs = get_random_graphs(500, 20, 0.2)
dense_graphs = get_random_graphs(500, 20, 0.4)
graphs = sparse_graphs + dense_graphs
graphs = np.stack(networkx_to_dense(graphs))
graphs = np.repeat(graphs[:, :, :, np.newaxis], 2, axis=3)

# create labels
sparse_labels = [0] * 500
dense_labels = [1] * 500
labels = sparse_labels + dense_labels

# visualize the edge count for the graphs
# plot_nx_edge_count(sparse_graphs, dense_graphs)

# Design your Pipeline
my_pipe = Hyperpipe('basic_gcn_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=5),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['accuracy', 'balanced_accuracy', 'recall', 'precision'],
                    best_config_metric='accuracy')

my_pipe.add(PipelineElement('GraphConstructorThreshold', threshold=0.95))

my_pipe.add(PipelineElement('GCNClassifier', feature_axis=0, allow_zero_in_degree=True))

my_pipe.fit(graphs, labels)
