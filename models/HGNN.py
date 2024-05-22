
import copy
import os
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from models.basic_block import RBF
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data

class SelfAttention(nn.Module):
    """
    The class is an implementation of the multi-head self attention
    "A Structured Self-Attentive Sentence Embedding including regularization"
    https://arxiv.org/abs/1703.03130 in ICLR 2017
    We made light modifications for speedup
    """

    def __init__(self, hidden):
        super().__init__()

        self.first_linear = nn.Linear(hidden, 16)
        self.first_linear.bias.data.fill_(0)
        self.second_linear = nn.Linear(16, 1)
        self.second_linear.bias.data.fill_(0)

    def forward(self, encoder_outputs):

        # (B, Length, H) -> (B , Length, 10)
        first_hidden = self.first_linear(encoder_outputs)
        energy = self.second_linear(torch.tanh(first_hidden))

        attention = F.softmax(energy, dim=1).transpose(1, 2)  # (B, 10, Length)
        # encoder_outputs is (B, Length, Hidden)
        sentence_embeddings = attention @ encoder_outputs
        outputs = sentence_embeddings.sum(dim=1)
        return outputs


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, embed_dim, rbf_params=None, device=None):
        super(BondFloatRBF, self).__init__()

        if rbf_params is None:
            self.rbf_params = (torch.arange(0, 2, 0.1).to(device), 10.0)  # (centers, gamma)
        else:
            self.rbf_params = rbf_params

        centers, gamma = self.rbf_params
        self.rbf = RBF(centers, gamma).to(device)
        self.linear = nn.Linear(len(centers), embed_dim).to(device)

    def forward(self, x):
        """
        Args:
            edge_float_features(dict of tensor): edge float features.
        """
        out_embed = 0
        rbf_x = self.rbf(x)
        out_embed += self.linear(rbf_x)
        return out_embed


class HMRGNNConv(MessagePassing):
    def __init__(self, input_dim, output_dim, update_mode, norm=False, batch_norm=False, dropout=0.,
                 aggr='mean', device=None):
        super(HMRGNNConv, self).__init__(aggr=aggr)

        self.update_mode = update_mode
        self.concat_linear = nn.Linear(input_dim * 2, output_dim).to(device)
        self.root_linear = nn.Linear(input_dim, output_dim).to(device)
        self.norm = norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_embeddings):
        """
        :param x: [num_node, d]
        :param edge_index: [2, num_edge]
        :param edge_attr: [num_edge, num_attr]
        :param edge_weight: [num_edge, 1]
        :return:
        """

        msg = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        if self.update_mode:
            msg += self.root_linear(x)
        else:
            msg = self.root_linear(x)
        if self.norm:
            msg = F.normalize(msg, p=2, dim=-1)
        if self.batch_norm:
            msg = self.batch_norm(msg)
        return self.dropout(self.act(msg))

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class HMRGNN(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, num_layers, update_mode, num_edge_type, num_edge_attr_int,
                 num_edge_attr_float, num_relation_attr, batch_norm=False, dropout=0., device=None):
        super(HMRGNN, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.relation_type_encoder = nn.Embedding(num_edge_type+1, hidden_dim).to(device)
        self.int_relation_attr_encoder = nn.ModuleList([nn.Embedding(num_embeddings=num_relation_attr,
                                                                     embedding_dim=hidden_dim, device=device)
                                                       for _ in range(num_edge_attr_int)]).to(device)
        self.float_relation_attr_encoder = nn.ModuleList([BondFloatRBF(hidden_dim, device=device)
                                                         for _ in range(num_edge_attr_float)]).to(device)
        self.edge_embedding_layer = nn.Linear(hidden_dim*2, hidden_dim).to(device)
        self.GNN_blocks = nn.ModuleList()
        for i in range(num_edge_type-1):
            for j in range(num_layers):
                self.GNN_blocks.append(HMRGNNConv(input_dim=hidden_dim, output_dim=hidden_dim, update_mode=True,
                                                  norm=True, dropout=dropout, device=device))
        self.update_mode = update_mode
        self.concat_linear = nn.Linear(hidden_dim*2, output_dim).to(device)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim).to(device)
        else:
            self.batch_norm = None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()


    def forward(self, x, edge_index, edge_type, edge_type_list, edge_weight):
        """
        :param x: [num_node, d]
        :param edge_index: [2, num_edge]
        :param edge_type: [num_edge, num_attr]
        :param edge_weight: [num_edge, 1]
        :return:
        """

        prop_edge_select = ~(edge_type_list-edge_type[0]).bool()
        sca_edge_select = ~(edge_type_list - edge_type[1]).bool()
        group_edge_select = ~(edge_type_list-edge_type[2]).bool()
        x_edge_type = self.relation_type_encoder(edge_type_list)
        x_edge_attr = x_edge_type.clone().detach()
        x_edge_attr[prop_edge_select, :] = self.int_relation_attr_encoder[0](edge_weight[prop_edge_select].long())
        x_edge_attr[sca_edge_select, :] = self.float_relation_attr_encoder[0](edge_weight[sca_edge_select])
        x_edge_attr[group_edge_select, :] = self.float_relation_attr_encoder[1](edge_weight[group_edge_select])
        x_edge = self.edge_embedding_layer(torch.cat([x_edge_type, x_edge_attr], -1))
        x_edge = F.normalize(x_edge, p=2, dim=-1)
        x_edge = self.dropout(self.act(x_edge))
        x_s = x.clone()
        x_g = x.clone()
        for i in range(self.num_layers):
            x_s = self.GNN_blocks[i](x=x_s, edge_index=edge_index[:, prop_edge_select+sca_edge_select],
                                     edge_embeddings=x_edge[prop_edge_select+sca_edge_select])
            x_g = self.GNN_blocks[i+len(edge_type)-1](x=x_g, edge_index=edge_index[:, prop_edge_select+group_edge_select],
                                                      edge_embeddings=x_edge[prop_edge_select+group_edge_select])
        x_mol = torch.cat((x_s, x_g), dim=1)
        x_mol = self.concat_linear(x_mol)
        if self.batch_norm:
            x_mol = self.batch_norm(x_mol)
        x_mol = self.dropout(self.act(x_mol))
        return x_mol

