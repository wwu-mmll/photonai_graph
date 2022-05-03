from sklearn.base import BaseEstimator, ClassifierMixin
from photonai_graph.util import assert_imported
import numpy as np
try:
    from grakel import graph
    import grakel
except ImportError:  # pragma: no cover
    pass


class GrakelAdapter(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 input_type: str = "dense",
                 node_labels: dict = None,
                 edge_labels: dict = None,
                 node_feature_construction: str = "mean",
                 feature_axis: int = 1,
                 adjacency_axis: int = 0):
        """
        A transformer class for transforming graphs into Grakel format.


        Parameters
        ----------
        input_type: str,default="dense"
            the type of the input, dense or networkx
        node_labels: list,default=None
            a list of the node labels, needed for shortest path kernel
        edge_labels: dict,default=None
            list of edge labels if graphs are constructed from networkx graphs
        node_feature_construction: str,default="mean"
            mode of feature construction for graphs constructed from adjacency matrices. "mean" takes the mean of the
            nodes edge weights, "sum" takes the sum of the nodes edge weights, "degree" takes the node degree,
             and "features" takes the nodes features as supplied in the feature matrix.
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        feature_axis: int,default=1
            position of the feature axis, default being 1


        Example
        ----------
        ```python
        adapter = GrakelAdapter(input_type="dense", node_feature_construction="features")
        ```
        """
        self.input_type = input_type
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.node_feature_construction = node_feature_construction
        self.feature_axis = feature_axis
        self.adjacency_axis = adjacency_axis
        assert_imported(["grakel"])
        if input_type not in ['networkx', 'dense']:
            raise ValueError("Only networkx or dense conversions are supported.")

    def fit(self, X, y):
        return self

    def transform(self, X):
        """sklearn compatible graph conversion"""
        if self.input_type == "dense":
            node_features = self.construct_node_features(X)
            edge_features = self.construct_edge_features(X)
        else:
            node_features = None
            edge_features = None
        X_transformed = self.convert_grakel(X, self.input_type, node_features, edge_features, self.adjacency_axis)

        return X_transformed

    @staticmethod
    def convert_grakel(graphs, in_format, node_labels, edge_features, adjacency_axis):
        """convert graphs into grakel format"""
        g_trans = []
        if in_format == "dense":
            for g in range(graphs.shape[0]):
                conv_g = graph.Graph(graphs[g, :, :, adjacency_axis], node_labels=node_labels[g],
                                     edge_labels=edge_features[g])
                g_trans.append(conv_g)

        if in_format == "networkx":
            g_trans = grakel.graph_from_networkx(graphs)

        return g_trans

    def construct_node_features(self, matrices):
        """construct node features from the feature matrix"""
        label_list = []
        for mtrx in range(matrices.shape[0]):
            feat = self.get_dense_feature(matrices[mtrx, :, :, :], adjacency_axis=self.adjacency_axis,
                                          feature_axis=self.feature_axis, aggregation=self.node_feature_construction)
            label_list.append(feat)

        return label_list

    def construct_edge_features(self, matrices):
        """construct edge features from the feature or adjacency matrix"""
        label_list = []
        for mtrx in range(matrices.shape[0]):
            feat = self.get_dense_edge_features(matrices[mtrx, :, :, :], adjacency_axis=self.adjacency_axis,
                                                feature_axis=self.feature_axis)
            label_list.append(feat)

        return label_list

    @staticmethod
    def get_dense_edge_features(matrix, adjacency_axis, feature_axis):
        """returns the features for an edge label dictionary
            Parameters
            ---------
            matrix: np.matrix/np.ndarray
                feature matrix
            adjacency_axis: int
                position of the adjacency matrix
            feature_axis: int
                position of the feature matrix
        """
        edge_feat = {}
        for index, value in np.ndenumerate(matrix[:, :, adjacency_axis]):
            conn_key = (str(index[0]), str(index[1]))
            key_val = {conn_key: value}
            edge_feat.update(key_val)
        return edge_feat

    @staticmethod
    def get_dense_feature(matrix, adjacency_axis, feature_axis, aggregation="sum"):
        """returns the features for a networkx graph
            Parameters
            ---------
            matrix: np.matrix/np.ndarray
                feature matrix
            adjacency_axis: int
                position of the adjacency matrix
            feature_axis: int
                position of the feature matrix
            aggregation:
                method of feature construction, sum gives a row-wise sum,
                "mean" gives a row-wise mean, "node_degree" give a row-wise node-degree,
                features returns the entire row as the feature vector
        """
        if aggregation == "sum":
            features = np.sum(matrix[:, :, feature_axis], axis=1)
            features = features.tolist()
        elif aggregation == "mean":
            features = (np.sum(matrix[:, :, feature_axis], axis=1)) / matrix.shape[0]
            features = features.tolist()
        elif aggregation == "node_degree":
            features = np.count_nonzero(matrix[:, :, adjacency_axis], axis=1, keepdims=False)
            features = features.tolist()
        elif aggregation == "features":
            features = matrix[:, :, feature_axis]
            features = features.reshape((features.shape[0], -1))
            features = features.tolist()
        else:
            raise KeyError('Only sum, mean, node_degree and all features are supported')

        features = dict(enumerate(features, 0))

        return features
