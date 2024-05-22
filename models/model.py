
import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from .base_encoder import GNN_Encoder
from .relation import TaskRelationNet

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, A, X):
        A_norm = A - 1e9 * torch.less_equal(A, 0.8)
        A_norm = A_norm.softmax(-1)

        return F.leaky_relu(A_norm.mm(self.fc(X)))


class KRGTS(nn.Module):
    def __init__(self, args, task_num, train_task_num, device):
        super(KRGTS, self).__init__()
        self.args = args
        self.device = device
        self.mol_encoder = GNN_Encoder(num_layer=args.mol_num_layer,
                                       emb_dim=args.emb_dim,
                                       JK=args.JK,
                                       drop_ratio=args.mol_dropout,
                                       graph_pooling=args.mol_graph_pooling,
                                       gnn_type=args.mol_gnn_type,
                                       batch_norm=args.mol_batch_norm,
                                       load_path=args.mol_pretrain_load_path).to(device)

        self.relation_net = TaskRelationNet(args, in_dim=args.emb_dim,
                                            num_layer=args.rel_layer,
                                            total_tasks=task_num,
                                            train_tasks=train_task_num,
                                            top_k=args.rel_top_k,
                                            dropout=args.rel_dropout,
                                            pre_dropout=args.rel_pre_dropout,
                                            device=device, num_relation_attr=args.num_relation_attr)

    def encode_mol(self, data):
        return self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)

    def data_prepare(self, s_data, q_data, sca_adj, gro_adj, tasks):
        s_feat, q_feat = self.encode_mol(s_data), self.encode_mol(q_data)
        s_feat = s_feat.view(len(tasks), -1, s_feat.shape[1]).unsqueeze(1).expand(-1, len(q_data) // len(tasks), -1, -1)
        s_feat = s_feat.contiguous().view(-1, s_feat.shape[-2], s_feat.shape[-1])
        q_feat = q_feat.unsqueeze(1)

        X = torch.cat((s_feat, q_feat), 1)
        s_y = s_data.y[torch.arange(len(s_data)).unsqueeze(1), tasks.repeat_interleave(len(s_data) // len(tasks), 0)]
        q_y = q_data.y[torch.arange(len(q_data)).unsqueeze(1),
        tasks[:, 1:].repeat_interleave(len(q_data) // len(tasks), 0)]
        s_y = s_y.view(len(tasks), -1).repeat_interleave(len(q_data) // len(tasks), 0)

        # connect task-molecules
        s_nan = s_y.isnan()
        q_nan = q_y.isnan()
        edge_w_nan_index = torch.cat((s_nan, q_nan), -1).repeat(1, 2)
        edge_w_ls_temp = torch.cat((s_y, q_y), -1).repeat(1, 2)
        edge_w_ls_temp[edge_w_nan_index] = self.args.rel_nan_w
        edge_type_ls_temp = torch.zeros(edge_w_ls_temp.shape).to(self.device)
        s_index = (tasks.shape[1] + torch.arange(X.shape[1] - 1)).repeat_interleave(tasks.shape[1], -1)
        q_index = torch.tensor([tasks.shape[1] + X.shape[1] - 1]).repeat_interleave(tasks[:, 1:].shape[1], -1)
        mol_index = torch.cat((s_index, q_index), dim=-1).to(self.device)
        s_task_index = torch.arange(tasks.shape[1]).repeat(1, len(s_data) // len(tasks))[0]
        q_task_index = torch.arange(1, tasks.shape[1])
        task_index = torch.cat((s_task_index, q_task_index), dim=-1).to(self.device)
        edge_ls_temp = torch.hstack((torch.vstack((mol_index, task_index)), torch.vstack((task_index, mol_index))))
        edge_ls_temp = edge_ls_temp.unsqueeze(0).repeat_interleave(X.shape[0], 0)

        # connect relation graph
        edge_ls = None
        edge_w_ls = None
        edge_type_ls = None
        for i, (ls, w_ls, type_ls, adj1, adj2) in enumerate(zip(edge_ls_temp, edge_w_ls_temp, edge_type_ls_temp,
                                                                sca_adj, gro_adj)):
            # get the index
            indices1 = torch.nonzero(adj1)
            indices2 = torch.nonzero(adj2)
            # extract the value
            values1 = adj1[indices1[:, 0], indices1[:, 1]]
            values2 = adj1[indices2[:, 0], indices2[:, 1]]
            node_num = i * (tasks.shape[1] + adj1.shape[0])
            if edge_ls is None:
                edge_ls = torch.cat((ls, indices1.t() + tasks.shape[1], indices2.t() + tasks.shape[1]), dim=-1)
                edge_w_ls = torch.cat((w_ls, values1, values2), dim=-1)
                edge_type_ls = torch.cat((type_ls, torch.ones(values1.shape).to(self.device),
                                          torch.tensor([2] * values2.shape[0]).to(self.device)), dim=-1)
            else:

                edge_ls = torch.cat((edge_ls, ls + node_num, indices1.t() + tasks.shape[1] + node_num,
                                     indices2.t() + tasks.shape[1] + node_num), dim=-1)
                edge_w_ls = torch.cat((edge_w_ls, w_ls, values1, values2), dim=-1)
                edge_type_ls = torch.cat((edge_type_ls, type_ls, torch.ones(values1.shape).to(self.device),
                                          torch.tensor([2] * values2.shape[0]).to(self.device)), dim=-1)
        return X, edge_ls, edge_w_ls, edge_type_ls, tasks

    def get_embedding(self, s_data, q_data, sca_adj, gro_adj, tasks):
        X, edge_ls, edge_w_ls, edge_type_ls, tasks = self.data_prepare(s_data, q_data, sca_adj, gro_adj, tasks)
        graph_emb = self.relation_net.forward_subgraph_embedding(X, edge_ls, edge_w_ls, edge_type_ls, tasks)
        return graph_emb


    def forward(self, s_data, q_data, sca_adj, gro_adj, tasks):
        X, edge_ls, edge_w_ls, edge_type_ls, tasks = self.data_prepare(s_data, q_data, sca_adj, gro_adj, tasks)
        target_s_idx = torch.LongTensor([list(range(tasks.shape[1], tasks.shape[1] + len(s_data) // len(tasks))),
                                         [0] * (len(s_data) // len(tasks))])
        target_q_idx = torch.LongTensor([[tasks.shape[1] + len(s_data) // len(tasks)], [0]])
        support_logit, query_logit, graph_emb = self.relation_net.forward_inductive(X, edge_ls, edge_w_ls, edge_type_ls,
                                                                                    tasks, target_s_idx, target_q_idx,
                                                                                    len(s_data) // len(tasks),
                                                                                    len(q_data) // len(tasks))
        tgt_s_y = s_data.y[torch.arange(len(s_data)).unsqueeze(1), tasks[:, 0].unsqueeze(1).repeat_interleave(
            len(s_data) // len(tasks), 0)].view(len(tasks), -1).repeat_interleave(
            len(q_data) // len(tasks), 0).view(-1, 1)
        tgt_q_y = q_data.y[torch.arange(len(q_data)).unsqueeze(1), tasks[:, 0].unsqueeze(1).repeat_interleave(
            len(q_data) // len(tasks), 0)]

        return support_logit, query_logit, tgt_s_y, tgt_q_y, graph_emb

