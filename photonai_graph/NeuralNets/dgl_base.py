import warnings
from abc import ABC, abstractmethod
import os
from photonai_graph.util import assert_imported
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from tqdm import tqdm
try:
    import dgl
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dgl.dataloading import GraphDataLoader
except ImportError:
    import warnings
    warnings.warn("Some of the dependencies could not be loaded. Make sure to install all Dependencies:\n"
                  "https://wwu-mmll.github.io/photonai_graph/installation/#additional-packages")

from photonai_graph.GraphConversions import check_dgl
from photonai_graph.NeuralNets.NNUtilities import DGLData, zip_data


class DGLModel(BaseEstimator, ABC):
    _estimator_type = "classifier"

    def __init__(self, nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 verbose: bool = False,
                 logs: str = None):
        """
        Base class for DGL based graph neural networks. Implements
        helper functions and shared parameters used by other models.
        Implementation based on dgl python package.


        Parameters
        ----------
        nn_epochs: int,default=200
            the number of epochs which a model is trained
        learning_rate: float,default=0.001
            the learning rate when training the model
        batch_size: int,default=32
            number of samples per training batch
        adjacency_axis: int,default=0
            position of the adjacency matrix, default being zero
        feature_axis: int,default=1
            position of the feature matrix
        add_self_loops: bool,default=True
            self loops are added if true
        allow_zero_in_degree: bool,default=False
            If true the zero in degree test of dgl is disabled
        verbose: bool,default=False
            If true verbose information is printed
        logs: str,default=None
            Path to the log data

        """
        self.nn_epochs = nn_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adjacency_axis = adjacency_axis
        self.feature_axis = feature_axis
        self.add_self_loops = add_self_loops
        self.allow_zero_in_degree = allow_zero_in_degree
        if self.add_self_loops and self.allow_zero_in_degree:
            warnings.warn('If self loops are added allow_zero_in_degree should be false!')
        if not self.add_self_loops and not self.allow_zero_in_degree:
            warnings.warn('If no self loops are added and allow_zero_in_degree is set to false, '
                          'all graphs should not contain 0-in-degree nodes')
        self.model = None
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        assert_imported(["dgl", "pytorch"])
        self.verbose = verbose

    def train_model(self, epochs, model, optimizer, loss_func, data_loader):
        # This function trains the neural network
        epoch_losses = []
        for epoch in tqdm(range(epochs)):
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
            if self.verbose:
                print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

    def handle_inputs(self, x, adjacency_axis, feature_axis):
        """checks the format of the input and transforms them for dgl models"""
        x_trans = check_dgl(x, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        if self.add_self_loops:
            x_trans = [dgl.add_self_loop(x) for x in x_trans]
        return x_trans

    def fit(self, X, y=None):
        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        self._init_model(X, y)
        # get optimizers
        loss_func, optimizer = self.setup_model()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)
        return self

    def predict(self, x):
        return self.predict_model(x)

    @staticmethod
    @abstractmethod
    def collate(samples):
        """Collate function"""

    @abstractmethod
    def get_data_loader(self, x_trans, y):
        """Data loader"""

    @abstractmethod
    def predict_model(self, X):
        """get model predictions"""

    @abstractmethod
    def setup_model(self):
        """Setup the model"""

    @abstractmethod
    def _init_model(self, X=None, y=None):
        """initialize model"""


class DGLClassifierBaseModel(DGLModel, ClassifierMixin, ABC):
    def __init__(self,
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 logs: str = None):
        """Abstract base class for classification algorithms

                Parameters
                ----------
                nn_epochs: int,default=200
                    Number of epochs to fit the model
                learning_rate: float,default=0.001
                    Learning rate for model training
                batch_size: int,default=32
                    Batch size for model training
                adjacency_axis: int,default=0
                    Axis which contains the adjacency
                feature_axis: int,default=1
                    Axis which contains the features
                add_self_loops: bool,default=True
                    If this value is true, a self loop is added to each node of each graph
                allow_zero_in_degree: bool,default=False
                    If true the dgl model allows zero-in-degree Graphs
                logs: str,default=None
                    Default logging directory
                """
        super(DGLClassifierBaseModel, self).__init__(nn_epochs=nn_epochs,
                                                     learning_rate=learning_rate,
                                                     batch_size=batch_size,
                                                     adjacency_axis=adjacency_axis,
                                                     feature_axis=feature_axis,
                                                     add_self_loops=add_self_loops,
                                                     allow_zero_in_degree=allow_zero_in_degree,
                                                     logs=logs)

    def setup_model(self):
        """returns the loss and optimizer for classification"""
        loss_func = nn.CrossEntropyLoss()  # specify loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return loss_func, optimizer

    def predict_model(self, X):
        """returns the argmax of the predictions for classification tasks"""
        self.model.eval()
        x_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        test_bg = dgl.batch(x_trans)
        probs_y = torch.softmax(self.model(test_bg), 1)
        argmax_y = torch.max(probs_y, 1)[1].view(-1, 1)
        return argmax_y.squeeze()

    def get_data_loader(self, x_trans, y):
        """returns data in a data loader format"""
        data = DGLData(zip_data(x_trans, y))
        # create dataloader
        data_loader = GraphDataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        return data_loader

    @staticmethod
    def collate(samples):
        """returns a batched graph, the input (samples) is a list of pairs (graph, label)"""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.long)


class DGLRegressorBaseModel(DGLModel, RegressorMixin, ABC):
    def __init__(self,
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 logs: str = None):
        """Abstract base class for regression algorithms

        Parameters
        ----------
        nn_epochs: int,default=200
            Number of epochs to fit the model
        learning_rate: float,default=0.001
            Learning rate for model training
        batch_size: int,default=32
            Batch size for model training
        adjacency_axis: int,default=0
            Axis which contains the adjacency
        feature_axis: int,default=1
            Axis which contains the features
        add_self_loops: bool,default=True
            If this value is true, a self loop is added to each node of each graph
        allow_zero_in_degree: bool,default=False
            If true the dgl model allows zero-in-degree Graphs
        logs: str,default=None
            Default logging directory
        """
        super(DGLRegressorBaseModel, self).__init__(nn_epochs=nn_epochs,
                                                    learning_rate=learning_rate,
                                                    batch_size=batch_size,
                                                    adjacency_axis=adjacency_axis,
                                                    feature_axis=feature_axis,
                                                    add_self_loops=add_self_loops,
                                                    allow_zero_in_degree=allow_zero_in_degree,
                                                    logs=logs)

    def setup_model(self):
        """returns the loss and optimizer for regression"""
        loss_func = nn.MSELoss()  # regression loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return loss_func, optimizer

    def predict_model(self, X):
        self.model.eval()
        x_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        test_bg = dgl.batch(x_trans)
        probs = self.model(test_bg)
        probs = probs.detach().numpy()
        return probs.squeeze()

    def get_data_loader(self, x_trans, y):
        """returns data in a regression data loader format"""
        y = y.reshape(y.shape[0], 1)
        data = DGLData(zip_data(x_trans, y))
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        return data_loader

    @staticmethod
    def collate(samples):
        """returns a batched graph, the input (samples) is a list of pairs"""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.float32)
