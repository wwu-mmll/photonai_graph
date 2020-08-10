# Workflow

The starting point for PHOTON Graph can be either connectivity matrices or data that is already in a graph format (networkx, dgl, sparse/dense adjacency matrices). Depending on your starting point, there are different ways in which you will you will have set up your pipeline. In the case that you have connectivity matrices, you will need to use graph constructors to turn those connectivity matrices into adjacency matrices. After can choose from different options of doing machine learning with your graph data.

### Connectivity matrices

If you have connectivity matrices as your starting point, these might be noisy and densely connected, as for example in the case of resting state functional connectivity in the area of neuroscience. In order to reduce the amount connections and possibly reduce noise, one could threshold the graph so weak connections will be discarded. This is not the only possible way to construct adjacency matrices, and many more methods have been implemented, which can be found in the graph constructor section.

After transforming your matrix, using a graph constructor you can then use this matrix to do machine learning with it.

### Machine Learning on Graphs

Once you have a graph structure, you can then use this graph structure to do machine learning on it in a variety of ways. One option would be to extract graph measures and use these graph measures to do classical machine learning on them. The measures preserve graph information, that would be lost if only looking at node values for example. Depending on the measure it might contain global or local graph information. A similar idea applies to graph embeddings and kernels. They provide lower-dimensional representations of the graph structure, while still preserving graph information. The resulting embedding/kernel transformation can then be used to do classical machine learning.

In contrast Graph Neural Nets are modified neural networks, that learn directly on graphs and make use of graph information. Here different architectures are available, and no transformation step is required prior to the network. Similar to classical machine learning algorithms

### Building a pipeline

Before building a pipeline, one has to determine the starting point. For illustrative purposes we will consider the case where one is starting with connectivity matrices. First you pick a graph constructor to turn your matrices into adjacency matrices. Then one has to choose whether transform the graph into a low dimensional representation or to directly use graph neural networks. Using PHOTON Graph this can be done in just a few lines of code.

#### Example

```python
from photonai.base import Hyperpipe, PipelineElement
from photonai_graph.GraphUtilities import get_random_connectivity_data, get_random_labels
from sklearn.model_selection import KFold

# make random matrices to simulate connectivity matrices
X = get_random_connectivity_data(number_of_nodes=50, number_of_individuals=100)
y = get_random_labels(type="regression", number_of_labels=100)

# Design your Pipeline
my_pipe = Hyperpipe('basic_gembedding_pipe',
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
```

### Extended Notes

PHOTON Graph is written in such a way that each function is sklearn compatible. This means that every method can also used in custom pipelines outside of the PHOTON framework.
