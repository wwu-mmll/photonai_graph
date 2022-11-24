import numpy as np
try:
    import dgl
    import torch.nn as nn
    import torch.nn.functional as F
    from dgl.nn.pytorch import GraphConv
except ImportError:
    pass

from photonai_graph.NeuralNets.dgl_base import DGLRegressorBaseModel, DGLClassifierBaseModel


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, hidden_layers, allow_zero_in_degree):
        super(GCNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        # input layers
        self.layers.append(GraphConv(in_dim, hidden_dim, allow_zero_in_degree=allow_zero_in_degree))
        # hidden layers
        for layer in range(1, hidden_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=allow_zero_in_degree))
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        for i, gnn in enumerate(self.layers):
            h = F.relu(gnn(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


class GCNClassifierModel(DGLClassifierBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 *args,
                 **kwargs):
        """
        Graph Attention Network for graph classification. GCN Layers
        from Kipf & Welling, 2017.
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
        super(GCNClassifierModel, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

    def _init_model(self, X=None, y=None):
        self.model = GCNClassifier(self.in_dim,
                                   self.hidden_dim,
                                   len(np.unique(y)),
                                   self.hidden_layers,
                                   allow_zero_in_degree=self.allow_zero_in_degree)


class GCNRegressorModel(DGLRegressorBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 *args,
                 **kwargs):
        """
        Graph convolutional Network for graph regression. GCN Layers
        from Kipf & Welling, 2017. Implementation based on dgl & pytorch.


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
        super(GCNRegressorModel, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

    def _init_model(self, X=None, y=None):
        self.model = GCNClassifier(self.in_dim,
                                   self.hidden_dim, 1,
                                   self.hidden_layers,
                                   allow_zero_in_degree=self.allow_zero_in_degree).float()
