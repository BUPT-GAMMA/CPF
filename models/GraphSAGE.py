import dgl
import torch
import torch.nn as nn
from dgl.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
