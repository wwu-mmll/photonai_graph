from abc import ABC
import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from photonai_graph.GraphConversions import check_dgl
from photonai_graph.NeuralNets.NNUtilities import DGLData, zip_data
from sklearn.base import BaseEstimator, ClassifierMixin


class DGLmodel(BaseEstimator, ClassifierMixin, ABC):
    _estimator_type = "predictor"
    """
    Base class for DGL based graph neural networks. Implements 
    helper functions and shared paramtersused by other models. 
    Implementation based on gem python package.


    Parameters
    ----------
    * `nn_epochs` [int, default=200]:
        the number of epochs which a model is trained
    * `learning_rate` [float, default=0.001]:
        the learning rate when training the model
    * `batch_size` [int, default=32]:
        number of samples per training batch
    * `adjacency_axis` [int, default=0]:
        position of the adjacency matrix, default being zero
    * `feature_axis` [int, default=1]
        position of the feature matrix
    * `logs` [str, default=None]:
        Path to the log data

    """

    def __init__(self, nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 logs=''):
        self.nn_epochs = nn_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adjacency_axis = adjacency_axis
        self.feature_axis = feature_axis
        self.model = None
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
            iteration = 0
            for it, (bg, label) in enumerate(data_loader):
                prediction = model(bg)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                iteration = it
            epoch_loss /= (iteration + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

    @staticmethod
    def collate(samples):
        """returns a batched graph, the input (samples) is a list of pairs (graph, label)"""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def handle_inputs(x, adjacency_axis, feature_axis):
        """checks the format of the input and transforms them for dgl models"""
        x_trans = check_dgl(x, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        return x_trans

    def get_data_loader(self, x_trans, y):
        """returns data in a data loader format"""
        data = DGLData(zip_data(x_trans, y))
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)

        return data_loader

    def predict_classifier(self, x):
        """returns the argmax of the predictions for classification tasks"""
        self.model.eval()
        x_trans = self.handle_inputs(x, self.adjacency_axis, self.feature_axis)
        test_bg = dgl.batch(x_trans)
        probs_y = torch.softmax(self.model(test_bg), 1)
        argmax_y = torch.max(probs_y, 1)[1].view(-1, 1)

        return argmax_y

    def predict_regressor(self, x):
        """returns the predictions for a regression model"""
        self.model.eval()
        x_trans = self.handle_inputs(x, self.adjacency_axis, self.feature_axis)
        test_bg = dgl.batch(x_trans)

        return torch.softmax(self.model(test_bg), 1)

    def get_classifier(self):
        """returns the loss and optimizer for classification"""
        loss_func = nn.CrossEntropyLoss()  # specify loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return loss_func, optimizer

    def get_regressor(self):
        """returns the loss and optimizer for regression"""
        loss_func = nn.MSELoss()  # regression loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return loss_func, optimizer
