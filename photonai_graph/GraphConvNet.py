import numpy as np
import networkx
import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from photonai_graph.GraphConversions import check_dgl
from photonai_graph.NeuralNets.NNModels import GCNClassifier, GATClassifier
from photonai_graph.NeuralNets.NNUtilities import DGLData, zip_data
from sklearn.base import BaseEstimator, ClassifierMixin


# base class for all other DGL models
class DGLmodel(BaseEstimator, ClassifierMixin):
    # base class for other NN models based on dgl
    def __init__(self, nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 logs=''):
        self.nn_epochs = nn_epochs
        self.learning_rate = learning_rate
        self.adjacency_axis = adjacency_axis
        self.feature_axis = feature_axis
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    @staticmethod
    def train_model(epochs, model, optimizer, loss_func, data_loader):
        # This function trains the neural network
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for iter, (bg, label) in enumerate(data_loader):
                prediction = model(bg)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

    @staticmethod
    def collate(samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def handle_inputs(x, adjacency_axis, feature_axis):
        # this function checks what format the inputs have
        # and handles them
        x_trans = check_dgl(x, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        return x_trans


class GCNClassifierModel(DGLmodel):

    def __init__(self, learning_rate=0.001):
        super().__init__()

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # prepare input data
        data = DGLData(zip_data(X_trans, y))
        # instantiate DataLoader
        data_loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=self.collate)
        # specify model with optimizer etc
        self.model = GCNClassifier(1, 256, len(np.unique(y)))  # set model class (import from NN.models)
        loss_func = nn.CrossEntropyLoss()  # specify loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # specify optimizer
        self.model.train()  # train model

        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, X):

        self.model.eval()
        test_bg = dgl.batch(X)
        probs_Y = torch.softmax(self.model(test_bg), 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)

        return argmax_Y


class GATClassifierModel(DGLmodel):

    def __init__(self, nn_epochs=200, learning_rate=0.001):
        super().__init__()

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # prepare input data
        data = DGLData(zip_data(X_trans, y))
        # make data accessible
        data_loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=self.collate)
        # specify model with optimizer etc
        self.model = GATClassifier(1, 256, 2, len(np.unique(y)))  # set model class (import from NN.models)
        loss_func = nn.CrossEntropyLoss()  # specify loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # specify optimizer
        self.model.train()  # train model

        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, x):

        self.model.eval()
        x_trans = self.handle_inputs(x, self.adjacency_axis, self.feature_axis)
        test_bg = dgl.batch(x_trans)
        probs_y = torch.softmax(self.model(test_bg), 1)
        argmax_y = torch.max(probs_y, 1)[1].view(-1, 1)

        return argmax_y
