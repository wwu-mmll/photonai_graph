from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase
from gem.embedding.lap import LaplacianEigenmaps


class GraphEmbeddingLaplacianEigenmaps(GraphEmbeddingBase):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Laplacian Eigenmaps (Belkin & Niyogi, 2013).
    Implementation based on gem python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
        

    Example
    -------
        constructor = GraphEmbeddingLaplacianEigenmaps(embedding_dimension=1,
                                                       decay_factor=0.1)
    """

    def __init__(self, embedding_dimension: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dimension = embedding_dimension

    def fit(self, X, y):
        return self

    def transform(self, X):
        """Transforms graph using Laplacian Eigenmaps Embedding"""
        embedding = LaplacianEigenmaps(d=self.embedding_dimension)

        X_transformed = self.calculate_embedding(embedding, X)

        return X_transformed
