import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification, DeepGraphCNN
from stellargraph import StellarGraph
from sklearn import model_selection
from IPython.display import display, HTML
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.losses import binary_crossentropy, kullback_leibler_divergence, mae, mse
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from photonai.graph.base.GraphUtilities import DenseToNetworkx
from sklearn.base import BaseEstimator, ClassifierMixin


class GraphConvNet_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 GCN_layer_sizes: list = [64, 64],
                 learning_rate: float = 0.005,
                 loss: str = "binary_crossentropy",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int =64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='relu',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self.GCN_layer_sizes = GCN_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']


    def fit(self, X, y):

        # encode targets the right way
        graph_labels = self.encode_targets(y)
        graph_labels = pd.get_dummies(graph_labels, drop_first=True)
        # transform inputs
        X_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in X_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)

        # instantiate generator
        generator = PaddedGraphGenerator(graphs=graphs)

        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

        test_accs = []

        stratified_folds = model_selection.RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.n_repeats).split(graph_labels, graph_labels)

        for i, (train_index, test_index) in enumerate(stratified_folds):
            print(f"Training and evaluating on fold {i + 1} out of {self.folds * self.n_repeats}...")
            train_gen, test_gen = self.get_generators(train_index, test_index, graph_labels, batch_size=self.nn_batch_size, generator=generator)

            self.model = self.create_graph_classification_model(generator)

            history, acc = self.train_fold(self.model, train_gen, test_gen, es, self.epochs)

            test_accs.append(acc)

        return self

    def predict(self, X):
        return self.model.predict(X)


    def create_graph_classification_model(self, generator):
        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=self.GCN_layer_sizes,
            activations=["relu"]*len(self.GCN_layer_sizes),
            generator=generator,
            dropout=self.dropout_rate,
        )
        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=16, activation="relu")(predictions)
        predictions = Dense(units=1, activation="sigmoid")(predictions)

        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=binary_crossentropy, metrics=["acc"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("acc")]

        return history, test_acc

    def get_generators(self, train_index, test_index, graph_labels, batch_size, generator):
        train_gen = generator.flow(train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
        test_gen = generator.flow(test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

        return train_gen, test_gen

    def encode_targets(self, y, type="sigmoid"):
        y_labels = y.copy()
        if type == "sigmoid":
            y_labels[y_labels == 0] = -1
            y_labels[y_labels == 1] = 1
        elif type == "multiclass":
            y_labels = y_labels

        return y_labels


class GraphConvNet_Regression(BaseEstimator, ClassifierMixin):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 GCN_layer_sizes: list = [64, 64],
                 learning_rate: float = 0.005,
                 loss: str = "mse",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int =64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='relu',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self.GCN_layer_sizes = GCN_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']


    def fit(self, X, y):

        # encode targets the right way
        graph_labels = pd.get_dummies(y, drop_first=True)
        # transform inputs
        X_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in X_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)

        # instantiate generator
        generator = PaddedGraphGenerator(graphs=graphs)

        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

        test_accs = []

        stratified_folds = model_selection.RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.n_repeats).split(graph_labels, graph_labels)

        for i, (train_index, test_index) in enumerate(stratified_folds):
            print(f"Training and evaluating on fold {i + 1} out of {self.folds * self.n_repeats}...")
            train_gen, test_gen = self.get_generators(train_index, test_index, graph_labels, batch_size=self.nn_batch_size, generator=generator)

            self.model = self.create_graph_classification_model(generator)

            history, acc = self.train_fold(self.model, train_gen, test_gen, es, self.epochs)

            test_accs.append(acc)

        return self

    def predict(self, X):
        return self.model.predict(X)


    def create_graph_classification_model(self, generator):
        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=self.GCN_layer_sizes,
            activations=["relu"]*len(self.GCN_layer_sizes),
            generator=generator,
            dropout=self.dropout_rate,
        )
        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=16, activation="relu")(predictions)
        predictions = Dense(units=1)(predictions)

        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=mse, metrics=["mae", "mse"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("mae")]

        return history, test_acc

    def get_generators(self, train_index, test_index, graph_labels, batch_size, generator):
        train_gen = generator.flow(train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
        test_gen = generator.flow(test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

        return train_gen, test_gen




# Deep Graph CNN Neural Network for Graph Classification
class DeepGraphCNN_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 GCN_layer_sizes: list = [32, 32, 32, 1],
                 learning_rate: float = 0.005,
                 loss: str = "binary_crossentropy",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int =64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='tanh',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self.GCN_layer_sizes = GCN_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate
        self.activations = activations

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']


    def fit(self, X, y):

        # encode targets the right way
        graph_labels = self.encode_targets(y)
        graph_labels = pd.get_dummies(graph_labels, drop_first=True)
        # transform inputs
        X_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in X_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)

        # instantiate generator
        generator = PaddedGraphGenerator(graphs=graphs)

        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

        test_accs = []

        stratified_folds = model_selection.RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.n_repeats).split(graph_labels, graph_labels)

        for i, (train_index, test_index) in enumerate(stratified_folds):
            print(f"Training and evaluating on fold {i + 1} out of {self.folds * self.n_repeats}...")
            train_gen, test_gen = self.get_generators(train_index, test_index, graph_labels, batch_size=self.nn_batch_size, generator=generator)

            self.model = self.create_graph_classification_model(generator)

            history, acc = self.train_fold(self.model, train_gen, test_gen, es, self.epochs)

            test_accs.append(acc)

        return self

    def predict(self, X):
        return self.model.predict(X)


    def create_graph_classification_model(self, generator):
        model = DeepGraphCNN(
            layer_sizes=self.GCN_layer_sizes,
            activations=[self.activations]*len(self.GCN_layer_sizes),
            generator=generator,
            k=30
        )
        x_inp, x_out = model.in_out_tensors()

        x_out = Conv1D(filters=16, kernel_size=97, strides=97)(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=self.dropout_rate)(x_out)
        predictions = Dense(units=1, activation="sigmoid")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=binary_crossentropy, metrics=["acc"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("acc")]

        return history, test_acc

    def get_generators(self, train_index, test_index, graph_labels, batch_size, generator):
        train_gen = generator.flow(train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
        test_gen = generator.flow(test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

        return train_gen, test_gen

    def encode_targets(self, y, type="sigmoid"):
        y_labels = y.copy()
        if type == "sigmoid":
            y_labels[y_labels == 0] = -1
            y_labels[y_labels == 1] = 1
        elif type == "multiclass":
            y_labels = y_labels

        return y_labels


class DeepGraphCNN_Regressor(BaseEstimator, ClassifierMixin):

    def __init__(self, multi_class: bool = True,
                 hidden_layer_sizes: list = None,
                 GCN_layer_sizes: list = [32, 32, 32, 1],
                 learning_rate: float = 0.005,
                 loss: str = "mse",
                 epochs: int = 200,
                 folds: int = 10,
                 n_repeats: int = 5,
                 nn_batch_size: int =64,
                 metrics: list = None,
                 callbacks: list = None,
                 verbosity=1,
                 dropout_rate=0.5,  # list or float
                 activations='tanh',  # list or str
                 optimizer="adam"):  # list or keras.optimizer

        self.GCN_layer_sizes = GCN_layer_sizes
        self.learning_rate = learning_rate
        self._loss = ""
        self._multi_class = None
        self.loss = loss
        self.multi_class = multi_class
        self.epochs =epochs
        self.folds = folds
        self.n_repeats = n_repeats
        self.nn_batch_size = nn_batch_size
        self.dropout_rate = dropout_rate
        self.activations = activations

        if callbacks:
            self.callbacks = callbacks
        else:
            self.callbacks = []

        if not metrics:
            metrics = ['accuracy']


    def fit(self, X, y):

        # encode targets the right way
        graph_labels = pd.get_dummies(y, drop_first=True)
        # transform inputs
        X_graphs = DenseToNetworkx(X)
        graphs = []
        for graph in X_graphs:
            graph = StellarGraph.from_networkx(graph, node_features="collapsed_weight")
            graphs.append(graph)

        # instantiate generator
        generator = PaddedGraphGenerator(graphs=graphs)

        es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

        test_accs = []

        stratified_folds = model_selection.RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.n_repeats).split(graph_labels, graph_labels)

        for i, (train_index, test_index) in enumerate(stratified_folds):
            print(f"Training and evaluating on fold {i + 1} out of {self.folds * self.n_repeats}...")
            train_gen, test_gen = self.get_generators(train_index, test_index, graph_labels, batch_size=self.nn_batch_size, generator=generator)

            self.model = self.create_graph_classification_model(generator)

            history, acc = self.train_fold(self.model, train_gen, test_gen, es, self.epochs)

            test_accs.append(acc)

        return self

    def predict(self, X):
        return self.model.predict(X)


    def create_graph_classification_model(self, generator):
        model = DeepGraphCNN(
            layer_sizes=self.GCN_layer_sizes,
            activations=[self.activations]*len(self.GCN_layer_sizes),
            generator=generator,
            k=30
        )
        x_inp, x_out = model.in_out_tensors()

        x_out = Conv1D(filters=16, kernel_size=97, strides=97)(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=self.dropout_rate)(x_out)
        predictions = Dense(units=1)(x_out)

        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(self.learning_rate), loss=mse, metrics=["mae", "mse"])

        return model

    def train_fold(self, model, train_gen, test_gen, es, epochs):
        history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("mae")]

        return history, test_acc

    def get_generators(self, train_index, test_index, graph_labels, batch_size, generator):
        train_gen = generator.flow(train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
        test_gen = generator.flow(test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

        return train_gen, test_gen
