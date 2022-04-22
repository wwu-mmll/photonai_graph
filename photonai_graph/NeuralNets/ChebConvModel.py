from typing import List

try:
    import dgl
    import torch.optim as optim
    import torch.nn as nn
    #from torch_geometric.nn import ChebConv, global_mean_pool
    from dgl.nn.pytorch.conv import ChebConv
    import torch.nn.functional as func
except ImportError:
    pass

from photonai_graph.NeuralNets.dgl_base import DGLClassifierBaseModel, DGLRegressorBaseModel


class ChebConvModel(nn.Module):
    """ChebConv model(network architecture can be modified)"""

    def __init__(self,
                 num_features,
                 num_classes,
                 k_order,
                 dropout=.3):
        super(ChebConvModel, self).__init__()

        self.p = dropout

        self.conv1 = ChebConv(int(num_features), 128, k=k_order)
        self.conv2 = ChebConv(128, 64, k=k_order)
        self.conv3 = ChebConv(64, 32, k=k_order)

        self.lin1 = nn.Linear(32, int(num_classes))

    def forward(self, x):

        h = dgl.readout_nodes(x, 'feat').view(-1, 1).float()#  x.in_degrees().view(-1, 1).float()

        h = self.conv1(x, h)
        h = func.dropout(h, p=self.p, training=self.training)
        h = self.conv2(x, h)
        h = func.dropout(h, p=self.p, training=self.training)
        h = self.conv3(x, h)

        h = h.flatten(1)
        x.ndata['h'] = h
        hg = dgl.mean_nodes(x, 'h')
        return self.lin1(hg)
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #batch = data.batch

        #x = func.relu(self.conv1(x, edge_index, edge_attr))
        #x = func.dropout(x, p=self.p, training=self.training)
        #x = func.relu(self.conv2(x, edge_index, edge_attr))
        #x = func.dropout(x, p=self.p, training=self.training)
        #x = func.relu(self.conv3(x, edge_index, edge_attr))

        #x = global_mean_pool(x, batch)
        #x = self.lin1(x)
        # return x


class ChebConvClassifierModel(DGLClassifierBaseModel):

    def __init__(self,
                 in_dim: int = 1,
                 hidden_layers: int = 2,
                 hidden_dim: int = 256,
                 heads: List = None,
                 agg_mode="mean",
                 nn_epochs: int = 200,
                 learning_rate: float = 0.001,
                 model_batch_size: int = 16,
                 adjacency_axis: int = 0,
                 feature_axis: int = 1,
                 add_self_loops: bool = True,
                 allow_zero_in_degree: bool = False,
                 logs: str = ''):
        """
            Documentation todo


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
        super(ChebConvClassifierModel, self).__init__(nn_epochs=nn_epochs,
                                                      learning_rate=learning_rate,
                                                      batch_size=model_batch_size,
                                                      adjacency_axis=adjacency_axis,
                                                      feature_axis=feature_axis,
                                                      add_self_loops=add_self_loops,
                                                      allow_zero_in_degree=allow_zero_in_degree,
                                                      logs=logs)
        self.model_batch_size = model_batch_size
        if heads is None:
            heads = 2
            # Todo: if heads is not length of hidden layers +1 (bc of the first layer)
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.agg_mode = agg_mode

    def _init_model(self, X=None, y=None):
        self.model = ChebConvModel(self.in_dim, self.heads, 9)

    def setup_model(self):
        loss_func = nn.CrossEntropyLoss()  # specify loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return loss_func, optimizer
