import numpy as np
from photonai_graph.NeuralNets.dgl_base import DGLmodel
from photonai_graph.NeuralNets.NNModels import GATClassifier


class GATClassifierModel(DGLmodel):
    """
        Graph Attention Network for graph classification. GAT Layers
        are modeled after Veličković et al., 2018.
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
                 heads=None):
        super().__init__()
        if heads is None:
            heads = [2, 2]
            # Todo: if heads is not length of hidden layers +1 (bc of the first layer)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.heads = heads

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        self.model = GATClassifier(self.in_dim, self.hidden_dim, self.heads, len(np.unique(y)), self.hidden_layers)  # set model class (import from NN.models)
        # get optimizers
        loss_func, optimizer = self.get_classifier()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, x):

        return self.predict_classifier(x)
