import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import *


class GATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 agg_activation=F.elu):
        super(GATLayer, self).__init__()

        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax

        self.agg_activation=agg_activation

    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, feat, bg):
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        self.g = bg
        h = self.feat_drop(feat)
        # V x K x F', K number of heads, F' dim of transformed features
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)                              # K x V x F'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)      # V x K x 1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)      # V x K x 1
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax()
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        # 3. apply normalizer
        ret = self.g.ndata['ft']                                  # V x K x F'
        ret = ret.flatten(1)

        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        # Clean ndata and edata
        self.clean_data()

        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute un-normalized attention values from src and dst
        a = self.activation(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(attention)
