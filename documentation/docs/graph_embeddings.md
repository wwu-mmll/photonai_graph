# Graph Embeddings

Graph Embeddings are a way to learn a low dimensional representation of a graph. Through a graph embedding a graph can be represented in low dimensional form, while preserving graph information. This low-dimensional representation can then be used for training classic machine learning algorithms that would otherwise make no use of the graph information.

THe Graph Embeddings used by PHOTON Graph are static graph embeddings, based on the gem python package (link gem).

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
