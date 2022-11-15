import warnings
from abc import ABC, abstractmethod
import os
from photonai_graph.util import assert_imported
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
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
                 validation_score: bool = False,
                 early_stopping: bool = False,
                 es_patience: int = 10,
                 es_tolerance: int = 9,
                 es_delta: float = 0,
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
        validation_score: bool,default=False
            If true the input data is split into train and test (90%/10%).
            The testset is then used to get validation results during training
        early_stopping: bool, default=False
            If true then the loss over multiple iterations is evaluated to see
            whether early stopping should be called on the model
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
        self.validation_score = validation_score
        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_tolerance = es_tolerance
        self.es_delta = es_delta
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

    def train_model(self, epochs, model, optimizer, loss_func, data_loader, val_loader=None):
        # This function trains the neural network
        epoch_losses = []
        val_losses = []
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
            if self.verbose and not self.validation_score:
                print('Epoch {} \tloss {:.4f}'.format(epoch, epoch_loss))
            if self.verbose and self.validation_score:
                val_loss = 0
                for it, (bg, label) in enumerate(val_loader):
                    prediction = model(bg)
                    loss = loss_func(prediction, label)
                    val_loss += loss.detach().item()
                    iteration = it
                val_loss /= (iteration + 1)
                print(f'Epoch {epoch} \tloss {epoch_loss:.4f} \tval loss {val_loss:.4f}')
                val_losses.append(val_loss)
            epoch_losses.append(epoch_loss)
            convergence = self.check_val_loss_divergence(val_losses, epoch_losses,
                                                         self.es_patience, self.es_tolerance, self.es_delta)
            if not convergence:
                break

    def fit(self, X, y=None):
        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        val_loader = None
        if not self.validation_score:
            data_loader = self.get_data_loader(X_trans, y)
        else:
            print("10% of your train data is used as validation data!\ndisable with validation_score=False")
            X_train, X_val, y_train, y_val = train_test_split(X_trans, y, test_size=.1)
            data_loader = self.get_data_loader(X_train, y_train)
            val_loader = self.get_data_loader(X_val, y_val)
        # specify model with optimizer etc
        self._init_model(X, y)
        # get optimizers
        loss_func, optimizer = self.setup_model()
        # train model
        self.model.train()
        self.train_model(epochs=self.nn_epochs, model=self.model, optimizer=optimizer,
                         loss_func=loss_func, data_loader=data_loader, val_loader=val_loader)
        return self

    def handle_inputs(self, x, adjacency_axis, feature_axis):
        """checks the format of the input and transforms them for dgl models"""
        x_trans = check_dgl(x, adjacency_axis=adjacency_axis, feature_axis=feature_axis)
        if self.add_self_loops:
            x_trans = [dgl.add_self_loop(x) for x in x_trans]
        return x_trans

    def predict(self, x):
        return self.predict_model(x)

    def get_data_loader(self, x_trans, y):
        data = DGLData(zip_data(x_trans, y))
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        return data_loader

    @staticmethod
    def check_val_loss_divergence(val_loss_list: list, ep_loss_list: list,
                                  patience: int = 10, tolerance: int = 9,
                                  delta=0):
        if len(val_loss_list) < 10:
            convergence = True
        else:
            val_recent = val_loss_list[-patience:]
            ep_recent = ep_loss_list[-patience:]
            train_val_divergence = [val - ep for val, ep in zip(val_recent, ep_recent)]
            # trend = np.diff(train_val_divergence)
            counter = sum(i > delta for i in train_val_divergence)
            if counter >= tolerance:
                convergence = False
            else:
                convergence = True

        return convergence

    @staticmethod
    @abstractmethod
    def collate(samples):
        """Collate function"""

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
                 validation_score: bool = False,
                 verbose: bool = False,
                 logs: str = None,
                 **kwargs):
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
                validation_score: bool,default=False
                    It true the input data is split into train and test (90%/10%).
                    The testset is then used to get validation results during training
                verbose: bool,default=False
                    If true verbose output is generated
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
                                                     validation_score=validation_score,
                                                     verbose=verbose,
                                                     logs=logs,
                                                     **kwargs)

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
                 validation_score: bool = False,
                 verbose: bool = False,
                 logs: str = None,
                 **kwargs):
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
        validation_score: bool,default=False
            If true the input data is split into train and test (90%/10%).
            The testset is then used to get validation results during training
        verbose: bool,default=False
            If true verbose output is generated
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
                                                    validation_score=validation_score,
                                                    verbose=verbose,
                                                    logs=logs,
                                                    **kwargs)

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
        return super(DGLRegressorBaseModel, self).get_data_loader(x_trans=x_trans, y=y)

    @staticmethod
    def collate(samples):
        """returns a batched graph, the input (samples) is a list of pairs"""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels, dtype=torch.float32)
