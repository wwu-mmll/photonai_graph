# Graph Neural Networks

Graph Neural Networks are a class of neural networks, optimized for deep learning on graphs. They provide an end-to-end solution for machine learning on graphs, unlike graph kernels/embeddings where a transformation step is applied before a "classical" machine learning algorithm. As they have attracted more attention in the recent years, a range of different architectures for this class has sprung up ranging from the Graph Convolutional Network (GCN, Kipf et al. 2017) to Graph Attention Networks (citation needed). The different architectures learn from the graph and it's overall structure, making use of the graph information unlike classical neural networks.

The graph neural network module of PHOTONAI Graph provides a variety of customizable out-of-the-box graph neural networks. They can be instantiated in one line of code and easily integrate into PHOTONAI pipelines.

## Graph Neural Network Module

The Graph Neural Network module consists of three parts. The Layer Module, where different layers are implemented and the message-passing steps of these are defined. The Model module where the module is constructed as a class (see pytorch neural networks). And the GraphConvNet module which calls the models and implements fit and transform steps, making them sklearn conform. This module also handles data conversions, converting graphs to the right format for the networks, which are written in pytorch.

You can also write your own custom graph neural network architecture, and register them via the PHOTON register function (link here). When writing your own custom neural nets you are free to choose your own package, as long as they implement fit, transform and predict functions like the GraphConvNet module classes. These can also be used as a blueprint if you want to integrate your own graph neural network architectures into PHOTONAI.

## DglModel

Abstract base class for all dgl based graph neural networks. Implements shared functions like training and input handlers for both regression and classification models.

| Parameter | type | Description |
| -----     | ----- | ----- |
| nn_epochs | int, default=200 | the number of epochs which a model is trained |
| learning_rate | float, default=0.001 | the learning rate when training the model |
| batch_size | int, default=32 | number of samples per training batch |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| feature_axis | int, default=1 | position of the feature matrix |
| logs | str, default=None | path to the log data |

## GCNClassifierModel

Graph Attention Network for graph classification. GCN Layers from Kipf & Welling, 2017. Implementation based on dgl & pytorch.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions of the hidden layers |


## SGConvClassifierModel

Graph convolutional network for graph classification. Simple Graph convolutional layers from Wu, Felix, et al., 2018. Implementation based on dgl & pytorch.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions in the hidden layers |


## GATClassifierModel

Graph Attention Network for graph classification. GAT Layers are modeled after Veličković et al., 2018. Implementation based on dgl & pytorch.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions in the hidden layers |
| heads | list, default=None | list with number of heads per hidden layer |


## GCNRegressorModel

Graph convolutional Network for graph regression. GCN Layers from Kipf & Welling, 2017. Implementation based on dgl & pytorch.            

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions of the hidden layers |

## SGConvRegressorModel

Graph convolutional network for graph regression. Simple Graph convolutional layers from Wu, Felix, et al., 2018. Implementation based on dgl & pytorch.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions in the hidden layers |

## GATRegressorModel

Graph Attention Network for graph regression. GAT Layers are modeled after Veličković et al., 2018. Implementation based on dgl & pytorch.

| Parameter | type | Description |
| -----     | ----- | ----- |
| in_dim | int, default=1 | input dimension |
| hidden_layers | int, default=2 | number of hidden layers used by the model |
| hidden_dim | int, default=256 | dimensions in the hidden layers |
| heads | list, default=None | list with number of heads per hidden layer |
