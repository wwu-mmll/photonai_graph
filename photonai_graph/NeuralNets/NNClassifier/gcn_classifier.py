import numpy as np
import dgl
import torch
from photonai_graph.NeuralNets.dgl_base import DGLmodel
from photonai_graph.NeuralNets.NNModels import GCNClassifier


class GCNClassifierModel(DGLmodel):
    """
    Graph Attention Network for graph classification. GCN Layers
    from Kipf & Welling, 2017.
    Implementation based on dgl & pytorch.


    Parameters
    ----------
    * `in_dim` [int, default=1]:
        input dimension
    * `hidden_layers` [int, default=2]:
        number of hidden layers used by the model
    * `hidden_dim` [int, default=256]:
        dimensions in the hidden layers
    * `heads` [list, default=None]:
        list with number of heads per hidden layer

    """

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 logs: str = ''):
        super(GCNClassifierModel, self).__init__(nn_epochs=nn_epochs,
                                                 learning_rate=learning_rate,
                                                 batch_size=batch_size,
                                                 adjacency_axis=adjacency_axis,
                                                 feature_axis=feature_axis,
                                                 logs=logs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        self.model = GCNClassifier(self.in_dim, self.hidden_dim, len(np.unique(y)), self.hidden_layers)  # set model class (import from NN.models)
        # get optimizers
        loss_func, optimizer = self.get_classifier()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, X):

        # self.model.eval()
        # test_bg = dgl.batch(X)
        # probs_y = torch.softmax(self.model(test_bg), 1)
        # argmax_y = torch.max(probs_y, 1)[1].view(-1, 1)

        return self.predict_classifier(X)
