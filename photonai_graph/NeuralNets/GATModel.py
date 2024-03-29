from typing import List

import numpy as np
try:
    import dgl
    import torch.nn as nn
    from dgl.nn.pytorch import GATConv
except ImportError:
    pass

from photonai_graph.NeuralNets.dgl_base import DGLClassifierBaseModel, DGLRegressorBaseModel


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, hidden_layers,
                 agg_mode, allow_zero_in_degree):
        super(GATModel, self).__init__()

        self.agg_mode = agg_mode
        self.layers = nn.ModuleList()
        # append gat layer according to inputs
        self.layers.append(GATConv(in_dim, hidden_dim, num_heads[0],
                                   allow_zero_in_degree=allow_zero_in_degree))
        # hidden layers
        for layer in range(1, hidden_layers):
            self.layers.append(GATConv(
                hidden_dim, hidden_dim, num_heads[layer],
                allow_zero_in_degree=allow_zero_in_degree))
        # output layer
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, bg):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = bg.ndata['feat']
        for i, gnn in enumerate(self.layers):
            h = gnn(bg, h)
            if self.agg_mode == 'flatten':
                h = h.flatten(1)
            else:
                h = h.mean(1)

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)


class GATClassifierModel(DGLClassifierBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 heads: List = None,
                 agg_mode="mean",
                 nn_epochs: int = 200,
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
                 gpu: bool = False,
                 verbose: bool = False,
                 logs: str = '',
                 **kwargs):
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
            validation_score: bool,default=False
                If true the input data is split into train and test (90%/10%).
                The testset is then used to get validation results during training
            verbose: bool,default=False
                If true verbose output is generated
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
                                                 validation_score=validation_score,
                                                 early_stopping=early_stopping,
                                                 es_patience=es_patience,
                                                 es_tolerance=es_tolerance,
                                                 es_delta=es_delta,
                                                 gpu=gpu,
                                                 verbose=verbose,
                                                 logs=logs,
                                                 **kwargs)
        if heads is None:
            heads = [2, 2]
            # Todo: if heads is not length of hidden layers +1 (bc of the first layer)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.agg_mode = agg_mode
        self.gpu = gpu

    def _init_model(self, X=None, y=None):
        self.model = GATModel(X.shape[1] if isinstance(X, (np.ndarray, np.array)) else X[0].num_nodes(),
                              self.hidden_dim, self.heads,
                              len(np.unique(y)), self.hidden_layers, self.agg_mode,
                              allow_zero_in_degree=self.allow_zero_in_degree)


class GATRegressorModel(DGLRegressorBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 heads: List = None,
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 logs: str = None,
                 validation_score: bool = False,
                 early_stopping: bool = False,
                 es_patience: int = 10,
                 es_tolerance: int = 9,
                 es_delta: float = 0,
                 gpu: bool = False,
                 verbose: bool = False,
                 agg_mode: str = None,
                 **kwargs):
        """
            Graph Attention Network for graph regression. GAT Layers
            are modeled after Veličković et al., 2018. Implementation
            based on dgl & pytorch.


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
            validation_score: bool,default=False
                If true the input data is split into train and test (90%/10%).
                The testset is then used to get validation results during training
            gpu: bool, default=False
                If True the system should try using a gpu instead of cpu for training the model
            verbose: bool,default=False
                If true verbose output is generated
        """
        super(GATRegressorModel, self).__init__(nn_epochs=nn_epochs,
                                                learning_rate=learning_rate,
                                                batch_size=batch_size,
                                                adjacency_axis=adjacency_axis,
                                                feature_axis=feature_axis,
                                                add_self_loops=add_self_loops,
                                                allow_zero_in_degree=allow_zero_in_degree,
                                                validation_score=validation_score,
                                                early_stopping=early_stopping,
                                                es_patience=es_patience,
                                                es_tolerance=es_tolerance,
                                                es_delta=es_delta,
                                                gpu=gpu,
                                                verbose=verbose,
                                                logs=logs,
                                                **kwargs)
        if heads is None:
            heads = [2, 2]
            # Todo: if heads is not length of hidden layers +1 (bc of the first layer)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.agg_mode = agg_mode
        self.gpu = gpu

    def _init_model(self, X=None, y=None):
        self.model = GATModel(X.shape[1], self.hidden_dim, self.heads, 1, self.hidden_layers,
                              allow_zero_in_degree=self.allow_zero_in_degree, agg_mode=self.agg_mode).float()
