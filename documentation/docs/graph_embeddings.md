# Graph Embeddings

Graph Embeddings are a way to learn a low dimensional representation of a graph. Through a graph embedding a graph can be represented in low dimensional form, while preserving graph information. This low-dimensional representation can then be used for training classic machine learning algorithms that would otherwise make no use of the graph information.

The Graph Embeddings used by PHOTON Graph are static graph embeddings, based on the [gem python package](https://github.com/palash1992/GEM).


## Class - GraphEmbeddingGraphFactorization

Transformer class for calculating a Graph Embedding based on Graph Factorization (Ahmed et al., 2013). Graph Factorization factorizes the adjacency matrix with regularization. Implementation based on gem python package.

| Parameter | type | Description |
| -----     | ----- | ----- |
| embedding_dimension | int, default=1 | the number of dimensions that the final embedding will have |
| maximum_iterations | int, default=10000 | the number of iterations used in sgd, when learning the embedding |
| learning_rate | float, default=1 * 10 ** -4 | the learning rate of sgd |
| regularization_coefficient | float, default=1.0 | the regularization coefficient for regulating the magnitude of the weights |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | Path to the log data |

#### Example
    
```python
    constructor = GraphEmbeddingGraphFactorization(maximum_iterations=20000,
                            			           regularization_coefficient=0.5)
```





## Class - GraphEmbeddingHOPE

Transformer class for calculating a Graph Embedding based on Higher-order proximity preserved embedding (Mingdong et al., 2016). Implementation based on gem python package.

| Parameter | type | Description |
| -----     | ----- | ----- |
| embedding_dimension | int, default=1 | the number of dimensions that the final embedding will have |
| decay_factor | float, default=0.01 | the higher order coefficient beta |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | Path to the log data |

#### Example

```python
constructor = GraphEmbeddingHOPE(embedding_dimension=1,
                            	 decay_factor=0.1)
```





## Class - GraphEmbeddingLaplacianEigenmaps

Transformer class for calculating a Graph Embedding based on Laplacian Eigenmaps (Belkin & Niyogi, 2013). Implementation based on gem python package.

| Parameter | type | Description |
| -----     | ----- | ----- |
| embedding_dimension | int, default=1 | the number of dimensions that the final embedding will have |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | Path to the log data |

```python
constructor = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1)
```





## Class - GraphEmbeddingLocallyLinearEmbedding

Transformer class for calculating a Graph Embedding based on Locally Linear Embedding (Roweis & Saul, 2000). Implementation based on gem python package.

| Parameter | type | Description |
| -----     | ----- | ----- |
| embedding_dimension | int, default=1 | the number of dimensions that the final embedding will have |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | Path to the log data |

#### Example

```python
constructor = GraphEmbeddingLocallyLinearEmbedding(embedding_dimension=1)
```





## Class - GraphEmbeddingSDNE

Transformer class for calculating a Graph Embedding based on Structural Deep Network Embedding (Wang, Cui & Zhu, 2016). Implementation based on gem python package.

| Parameter | type | Description |
| -----     | ----- | ----- |
| embedding_dimension | int, default=1 | the number of dimensions that the final embedding will have |
| seen_edge_reconstruction_weight | int, default=5 | the penalty parameter beta in matrix B of the 2nd order objective |
| first_order_proximity_weight | float, default=1e-5 | the weighing hyperparameter alpha for the 1st order objective |
| lasso_regularization_coefficient | float, default=1e-6 | the L1 regularization coefficient |
| ridge_regression_coefficient | float, default=1e-6 | the L2 regularization coefficient |
| number_of_hidden_layers | int, default=3 | the number of hidden layers in the encoder/decoder |
| layer_sizes | int, default=[50, 15,] | the number of units per layer in the hidden layers of the encoder/decoder. Vector of length number_of_hidden_layers -1 |
| num_iterations | int, default=50 | the number of iterations with which to train the network |
| learning_rate | float, default=0.01 | the learning rate with which the network is trained |
| batch_size | int, default=500 | the batch size when training the algorithm |
| adjacency_axis | int, default=0 | position of the adjacency matrix, default being zero |
| verbosity | int, default=0 | The level of verbosity, 0 is least talkative and gives only warn and error, 1 gives adds info and 2 adds debug |
| logs | str, default=None | Path to the log data |

#### Example

```python
constructor = GraphEmbeddingSDNE(embedding_dimension=1,
                            	 seen_edge_reconstruction_weight=10,
					             first_order_proximity_weight=1e-4
					             num_hidden_layers=5,
					             layer_sizes=[50, 25, 20, 15,])
```
