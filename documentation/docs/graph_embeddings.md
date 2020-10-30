# Graph Embeddings

Graph Embeddings are a way to learn a low dimensional representation of a graph. Through a graph embedding a graph can be represented in low dimensional form, while preserving graph information. This low-dimensional representation can then be used for training classic machine learning algorithms that would otherwise make no use of the graph information.

The Graph Embeddings used by PHOTON Graph are static graph embeddings, based on the [gem python package](https://github.com/palash1992/GEM).




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
