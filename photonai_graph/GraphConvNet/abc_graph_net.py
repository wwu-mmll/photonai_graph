from abc import ABC
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.keras.callbacks import EarlyStopping
from photonai_graph.GraphUtilities import DenseToNetworkx
from stellargraph import StellarGraph
from sklearn import model_selection


class GraphNet(ABC, BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.model = None

    def predict(self, X):
        # todo: duplicated code
        # convert graphs to networkx in order to import them
        X_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in X_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)
        # instantiate graph generators
        generator = PaddedGraphGenerator(graphs=graphs)
        # make test set
        test_gen = generator.flow(graphs)

        return self.model.predict(test_gen)

    def fit(self, X, y):
        graph_labels = pd.get_dummies(y, drop_first=True)
        # transform inputs
        x_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in x_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)

        # instantiate generator
        generator = PaddedGraphGenerator(graphs=graphs)

        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

        test_accs = []

        self.model = self.create_graph_classification_model(generator)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(graphs, graph_labels)
        train_gen, val_gen = self.get_generators(X_train, X_test, y_train, y_test, self.nn_batch_size, generator)

        history, acc = self.train_fold(self.model, train_gen, val_gen, es, self.epochs)

        test_accs.append(acc)

        return self

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        raise NotImplementedError("Subclasses must implement this")
