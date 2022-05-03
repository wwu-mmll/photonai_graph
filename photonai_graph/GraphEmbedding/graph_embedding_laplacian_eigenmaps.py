from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase
try:
    from gem.embedding.lap import LaplacianEigenmaps
    from gem.embedding.static_graph_embedding import StaticGraphEmbedding
except ImportError:  # pragma: no cover
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

    def _init_embedding(self) -> StaticGraphEmbedding:
        return LaplacianEigenmaps(d=self.embedding_dimension)
