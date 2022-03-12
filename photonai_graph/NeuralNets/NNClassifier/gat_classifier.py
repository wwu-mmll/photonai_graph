import numpy as np
from typing import List
from photonai_graph.NeuralNets.dgl_base import DGLmodel
from photonai_graph.NeuralNets.NNModels import GATClassifier


class GATClassifierModel(DGLmodel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 heads: List= None,
                 agg_mode = "mean",
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 logs: str = ''):
        """
            Graph Attention Network for graph classification. GAT Layers
            are modeled after Veličković et al., 2018.
            Implementation based on dgl & pytorch.


            Parameters
            ----------
            in_dim: int,default=1
                input dimension
            hidden_layers: int,default=2
                number of hidden layers used by the model
            hidden_dim: int,default=256
                dimensions in the hidden layers
            heads: list,default=None
                list with number of heads per hidden layer
            agg_mode: str, default='mean'
                aggregation mode for the graph convolutional layers

        """
        super(GATClassifierModel, self).__init__(nn_epochs=nn_epochs,
                                                 learning_rate=learning_rate,
                                                 batch_size=batch_size,
                                                 adjacency_axis=adjacency_axis,
                                                 feature_axis=feature_axis,
                                                 add_self_loops=add_self_loops,
                                                 allow_zero_in_degree=allow_zero_in_degree,
                                                 logs=logs)
        if heads is None:
            heads = [2, 2]
            # Todo: if heads is not length of hidden layers +1 (bc of the first layer)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.agg_mode = agg_mode

    def fit(self, X, y):

        # handle inputs
        X_trans = self.handle_inputs(X, self.adjacency_axis, self.feature_axis)
        # get data loader
        data_loader = self.get_data_loader(X_trans, y)
        # specify model with optimizer etc
        # set model class (import from NN.models)
        self.model = GATClassifier(self.in_dim, self.hidden_dim, self.heads,
                                   len(np.unique(y)), self.hidden_layers, self.agg_mode,
                                   allow_zero_in_degree=self.allow_zero_in_degree)
        # get optimizers
        loss_func, optimizer = self.get_classifier()
        # train model
        self.model.train()
        self.train_model(self.nn_epochs, self.model, optimizer, loss_func, data_loader)

        return self

    def predict(self, x):

        return self.predict_classifier(x)
