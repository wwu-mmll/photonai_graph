from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase
import numpy as np
try:
    from gem.embedding.lap import LaplacianEigenmaps
except ImportError:
    pass


class GraphEmbeddingLaplacianEigenmaps(GraphEmbeddingBase):
    _estimator_type = "transformer"

    def __init__(self,
                 embedding_dimension: int = 1,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
        Transformer class for calculating a Graph Embedding
        based on Laplacian Eigenmaps (Belkin & Niyogi, 2013).
        Implementation based on gem python package.


        Parameters
        ----------
        embedding_dimension: int,default=1
            the number of dimensions that the final embedding will have
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        logs: str,default=None
            Path to the log data


        Example
        -------
        ```python
        constructor = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1)
        ```
        """
        super(GraphEmbeddingLaplacianEigenmaps, self).__init__(embedding_dimension=embedding_dimension,
                                                               adjacency_axis=adjacency_axis,
                                                               logs=logs)

    def fit(self, X, y):
        return self

    def transform(self, X):
        """Transforms graph using Laplacian Eigenmaps Embedding"""
        embedding = LaplacianEigenmaps(d=self.embedding_dimension)

        X_transformed = self.calculate_embedding(embedding, X)

        X_transformed = np.real(X_transformed)

        return X_transformed
