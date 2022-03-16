from networkx.generators import barabasi_albert_graph, erdos_renyi_graph
from sklearn.model_selection import KFold
from photonai_graph.GraphConversions import networkx_to_dense
import numpy as np

from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphConversions import convert_graphs

# create dataset of 1000 graphs
rr_graphs = [erdos_renyi_graph(20, .25) for _ in range(500)]
er_graphs = [erdos_renyi_graph(20, .2) for _ in range(500)]
graphs = rr_graphs + er_graphs

# we have to transform the networkx graphs into numpy graphs before passing them to photon
graphs = np.array(convert_graphs(graphs, output_format="dense"))
graphs = np.expand_dims(graphs, axis=-1)

# create labels
rr_labels = [0] * 500
er_labels = [1] * 500
labels = rr_labels + er_labels

# Design your Pipeline
my_pipe = Hyperpipe('networkx_example_pipe',
                    inner_cv=KFold(n_splits=2),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['accuracy', 'balanced_accuracy', 'recall', 'precision'],
                    best_config_metric='accuracy')

my_pipe.add(PipelineElement('GCNClassifier', feature_axis=0, allow_zero_in_degree=True))

my_pipe.fit(graphs, labels)
