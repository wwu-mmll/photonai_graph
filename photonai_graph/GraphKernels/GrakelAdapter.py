from sklearn.base import BaseEstimator, ClassifierMixin
from grakel import graph
import grakel


class GrakelAdapter(BaseEstimator, ClassifierMixin):

    """
    A transformer class for transforming graphs into Grakel format.


    Parameters
    ----------
    * `input_type` [str, default="dense"]
        the type of the input, dense or networkx
    * `node_labels` [list, default=None]
        a list of the node labels, needed for shortest path kernel
    * `adjacency_axis` [int, default=0]
        position of the adjacency matrix, default being zero


    Example
    ----------
        converter = GrakelAdpater(node_labels=list(range(X.shape[1])))
    """

    def __init__(self,
                 input_type: str = "dense",
                 node_labels: list = None,
                 adjacency_axis: int = 0):
        self.input_type = input_type
        self.node_labels = node_labels
        self.adjacency_axis = adjacency_axis

    def fit(self, X, y):
        return self

    def transform(self, X):
        """sklearn compatible graph conversion"""
        X_transformed = self.convert_grakel(X, self.input_type, self.node_labels, self.adjacency_axis)

        return X_transformed

    @staticmethod
    def convert_grakel(graphs, in_format, node_labels, adjacency_axis):
        """convert graphs into grakel format"""
        g_trans = []
        if in_format == "dense":
            for g in range(graphs.shape[0]):
                if graph.is_adjacency(graphs[g, :, :, adjacency_axis]):
                    conv_g = graph.Graph(graphs[g, :, :, adjacency_axis], node_labels)
                    g_trans.append(conv_g)
                else:
                    raise Exception("Adjacency needs to be grakel conform")
        elif in_format == "networkx":
            g_trans = grakel.graph_from_networkx(graphs)
        else:
            raise ValueError("Only networkx or dense conversions are supported.")

        return g_trans
