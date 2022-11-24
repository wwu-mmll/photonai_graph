import numpy as np
try:
    import dgl
    import torch.nn as nn
    from dgl.nn.pytorch import SGConv
except ImportError:
    pass

from photonai_graph.NeuralNets.dgl_base import DGLClassifierBaseModel, DGLRegressorBaseModel


class SGConvClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, hidden_layers, allow_zero_in_degree):
        super(SGConvClassifier, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SGConv(in_dim, hidden_dim))
        # hidden layers
        for lr in range(1, hidden_layers):
            self.layers.append(SGConv(hidden_dim, hidden_dim, allow_zero_in_degree=allow_zero_in_degree))
        # output layer
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, bg):
        h = bg.in_degrees().view(-1, 1).float()
        for lr, layer in enumerate(self.layers):
            h = layer(bg, h)
        bg.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(bg, 'h')

        return self.classify(hg)


class SGConvClassifierModel(DGLClassifierBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 *args,
                 **kwargs):
        """
        Graph convolutional network for graph classification. Simple Graph
        convolutional layers from Wu, Felix, et al., 2018.
        Implementation based on dgl & pytorch.


        Parameters
        ----------
        in_dim: int,default=1
            input dimension
        hidden_layers: int,default=2
            number of hidden layers used by the model
        hidden_dim: int,default=256
            dimensions in the hidden layers
        validation_score: bool,default=False
            If true the input data is split into train and test (90%/10%).
            The testset is then used to get validation results during training
        verbose: bool,default=False
            If true verbose output is generated
        """
        super(SGConvClassifierModel, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

    def _init_model(self, X=None, y=None):
        self.model = SGConvClassifier(self.in_dim, self.hidden_dim,
                                      len(np.unique(y)), self.hidden_layers,
                                      allow_zero_in_degree=self.allow_zero_in_degree)


class SGConvRegressorModel(DGLRegressorBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 *args,
                 **kwargs):
        """
        Graph convolutional network for graph regression. Simple Graph
        convolutional layers from Wu, Felix, et al., 2018. Implementation
        based on dgl & pytorch.


        Parameters
        ----------
        in_dim: int,default=1
            input dimension
        hidden_layers: int,default=2
            number of hidden layers used by the model
        hidden_dim: int,default=256
            dimensions in the hidden layers
        validation_score: bool,default=False
            If true the input data is split into train and test (90%/10%).
            The testset is then used to get validation results during training
        verbose: bool,default=False
            If true verbose output is generated
        """
        super(SGConvRegressorModel, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

    def _init_model(self, X=None, y=None):
        self.model = SGConvClassifier(self.in_dim, self.hidden_dim, 1, self.hidden_layers,
                                      allow_zero_in_degree=self.allow_zero_in_degree).float()
