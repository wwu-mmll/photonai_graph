import dgl
import torch
from photonai_graph.NeuralNets.dgl_base import DGLmodel
from photonai_graph.NeuralNets.NNModels import GCNClassifier


class GCNRegressorModel(DGLmodel):
    """
    Graph convolutional Network for graph regression. GCN Layers
    from Kipf & Welling, 2017. Implementation based on dgl & pytorch.


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
                 hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        self.model = GCNClassifier(self.in_dim, self.hidden_dim, 1, self.hidden_layers)
        # get optimizers
        loss_func, optimizer = self.get_regressor()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, X):

        self.model.eval()
        test_bg = dgl.batch(X)
        probs_y = torch.softmax(self.model(test_bg), 1)

        return probs_y
