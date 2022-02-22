from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase
try:
    from gem.embedding.lle import LocallyLinearEmbedding
except ImportError:
    pass


class GraphEmbeddingLocallyLinearEmbedding(GraphEmbeddingBase):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Locally Linear Embedding (Roweis & Saul,
    2000).
    Implementation based on gem python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have


    Example
    -------
        constructor = GraphEmbeddingLocallyLinearEmbedding(embedding_dimension=1)
    """

    def __init__(self,
                 embedding_dimension: int = 1,
                 adjacency_axis: int = 0,
                 logs: str = ''):
        super(GraphEmbeddingLocallyLinearEmbedding, self).__init__(embedding_dimension=embedding_dimension,
                                                                   adjacency_axis=adjacency_axis,
                                                                   logs=logs)

    def transform(self, X):
        """Transform graph using Locally Linear Embedding"""

        embedding = LocallyLinearEmbedding(d=self.embedding_dimension)

        X_transformed = self.calculate_embedding(embedding, X)

        return X_transformed
