import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SGConv, GATConv


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


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, hidden_layers,
                 agg_mode, allow_zero_in_degree):
        super(GATClassifier, self).__init__()

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
        h = bg.in_degrees().view(-1, 1).float()
        for i, gnn in enumerate(self.layers):
            h = gnn(bg, h)
            if self.agg_mode == 'flatten':
                h = h.flatten(1)
            else:
                h = h.mean(1)

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)


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
