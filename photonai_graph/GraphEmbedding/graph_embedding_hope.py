import numpy as np
try:
    from gem.embedding.hope import HOPE
    from gem.embedding.static_graph_embedding import StaticGraphEmbedding
except ImportError:  # pragma: no cover
    pass

from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase


class GraphEmbeddingHOPE(GraphEmbeddingBase):
    _estimator_type = "transformer"

    def __init__(self,
                 embedding_dimension: int = 1,
                 decay_factor: float = 0.01,
                 adjacency_axis: int = 0,
                 logs: str = None):
        """
        Transformer class for calculating a Graph Embedding
        based on Higher-order proximity preserved embedding
        (Mingdong et al., 2016).
        Implementation based on gem python package.


        Parameters
        ----------
        embedding_dimension: int,default=1
            the number of dimensions that the final embedding will have
        decay_factor: float,default=0.01
            the higher order coefficient beta
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        logs: str,default=None
            Path to the log data



        Example
        -------
            constructor = GraphEmbeddingHOPE(embedding_dimension=1,
                                             decay_factor=0.1)
        """
        super(GraphEmbeddingHOPE, self).__init__(embedding_dimension=embedding_dimension,
                                                 adjacency_axis=adjacency_axis,
                                                 logs=logs)
        self.decay_factor = decay_factor
        self.orig_transformed = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        x_transformed = super(GraphEmbeddingHOPE, self).transform(X)
        self.orig_transformed = x_transformed
        if self.embedding_dimension == 1:
            x_transformed = np.squeeze(np.reshape(x_transformed, (X.shape[0], -1, 1)))
        return x_transformed

    def _init_embedding(self) -> StaticGraphEmbedding:
        embedding_dimension = self.embedding_dimension
        if embedding_dimension == 1:
            embedding_dimension = 2
        return HOPE(d=embedding_dimension, beta=self.decay_factor)
