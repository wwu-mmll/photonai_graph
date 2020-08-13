from abc import ABC
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class GraphConstructorAdjacency(BaseEstimator, TransformerMixin):

    @staticmethod
    def adjacency(dist, idx):
        """Return the adjacency matrix of a kNN photonai_graph."""
        M, k = dist.shape
        assert M, k == idx.shape
        assert dist.min() >= 0

        # Weights.
        sigma2 = np.mean(dist[:, -1]) ** 2
        dist = np.exp(- dist ** 2 / sigma2)

        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M * k)
        V = dist.reshape(M * k)
        W = sparse.coo_matrix((V, (I, J)), shape=(M, M))

        # No self-connections.
        W.setdiag(0)

        # Non-directed photonai_graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10
        assert type(W) is sparse.csr.csr_matrix

        return W
