import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from photonai_graph.NeuralNets.NNLayers import GATLayer


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, hidden_layers):
        super(GCNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        # input layers
        self.layers.append(GraphConv(in_dim, hidden_dim))
        # hidden layers
        for layer in range(1, hidden_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
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
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, hidden_layers):
        super(GATClassifier, self).__init__()

        self.layers = nn.ModuleList()
        # append gat layer according to inputs
        self.layers.append(GATLayer(in_dim, hidden_dim, num_heads[0]))
        # hidden layers
        for layer in range(1, hidden_layers):
            self.gat_layers.append(GATLayer(
                hidden_dim * num_heads[layer - 1], hidden_dim, num_heads[layer]))
        # output layer
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, bg):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = bg.in_degrees().view(-1, 1).float()
        for i, gnn in enumerate(self.layers):
            h = gnn(h, bg)
        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)