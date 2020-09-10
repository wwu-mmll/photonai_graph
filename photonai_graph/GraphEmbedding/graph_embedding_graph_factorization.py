from photonai_graph.GraphEmbedding.graph_embedding_base import GraphEmbeddingBase
from gem.embedding.gf import GraphFactorization


class GraphEmbeddingGraphFactorization(GraphEmbeddingBase):
    _estimator_type = "transformer"

    """
    Transformer class for calculating a Graph Embedding
    based on Graph Factorization (Ahmed et al., 2013).
    Graph Factorization factorizes the adjacency matrix
    with regularization. Implementation based on gem
    python package.


    Parameters
    ----------
    * `embedding_dimension` [int, default=1]:
        the number of dimensions that the final embedding will have
    * `maximum_iterations` [int, default=10000]
        the number of iterations used in sgd, when learning the embedding
    * `learning_rate` [float, default=1 * 10 ** -4]
        the learning rate of sgd
    * `regularization_coefficient` [float, default=1.0]
        the regularization coefficient for regulating the magnitude of the weights


    Example
    -------
        constructor = GraphEmbeddingGraphFactorization(maximum_iterations=20000,
                                                        regularization_coefficient=0.5)
    """

    def __init__(self, embedding_dimension=1,
                 maximum_iterations=10000,
                 learning_rate=1 * 10 ** -4,
                 regularization_coefficient=1.0,
                 construction_axis=0,
                 adjacency_axis: int = 0,
                 logs: str = ''):
        super(GraphEmbeddingGraphFactorization, self).__init__(embedding_dimension=embedding_dimension,
                                                               adjacency_axis=adjacency_axis,
                                                               logs=logs)
        self.maximum_iterations = maximum_iterations
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.construction_axis = construction_axis

    def transform(self, X):
        """Embedds the graph using GraphFactorization"""
        embedding = GraphFactorization(d=self.embedding_dimension, max_iter=self.maximum_iterations,
                                       eta=self.learning_rate, regu=self.regularization_coefficient)

        X_transformed = self.calculate_embedding(embedding, X)

        return X_transformed
