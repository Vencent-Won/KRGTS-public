
import os
import copy
import torch
import torch.nn.functional as F
from torch import nn
from models.HGNN import HMRGNN

class TaskRelationNet(nn.Module):
    def __init__(self, args,
                 in_dim,
                 num_layer,
                 total_tasks,
                 train_tasks,
                 top_k=-1,
                 dropout=0.,
                 pre_dropout=0.,
                 device=None,
                 num_relation_attr=3,
                 learn_model='knowledge'):
        super(TaskRelationNet, self).__init__()
        self.args = args
        self.dropout = dropout
        self.total_tasks = total_tasks
        self.num_layer = num_layer
        self.top_k = top_k
        self.device = device
        self.rel_norm = self.args.rel_norm
        self.hidden_dim = self.args.rel_hidden_dim
        self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(device)
        if pre_dropout > 0:
            self.pre_dropout = nn.Dropout(pre_dropout)
        else:
            self.pre_dropout = None
        self.task_emb = nn.Embedding(total_tasks, in_dim).to(device)
        self.task_emb.weight.data[train_tasks:, :] = 0
        self.learn_model = learn_model
        self.GNN = HMRGNN(args=args, input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim,
                          num_layers=num_layer, device=device, num_relation_attr=num_relation_attr, update_mode=True,
                          num_edge_type=3, num_edge_attr_int=1, num_edge_attr_float=2,
                          batch_norm=args.rel_batch_norm, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU(),
                                        nn.Linear(self.hidden_dim, 1)).to(device)

    def get_task_emb(self, task_id):
        return self.task_emb(task_id)

    def forward(self):
        pass

    def forward_embedding(self, x, edge_index, edge_w, edge_type, tasks):
        """
        :param sample_emb: [n_q, n_s+1, d]
        :param task_id: [n_task]
        :param support_y: [n_s, n_task]
        :param query_y: [n_q, n_task]
        :param return_auxi
        :return: support_logit: [n_q, n_s]
                 query_logit: [n_q, 1]
                 tgt_s_y: [n_q, n_s]
                 tgt_q_y: [n_q, 1]
                 graph_emb: [d]
        """
        task_emb = self.task_emb(tasks)  # [n_task, d]
        x = self.fc1(x)
        task_emb = self.fc1(task_emb)
        if self.pre_dropout:
            x = self.pre_dropout(x)  # [n_q, n_s+1, d]
            task_emb = self.pre_dropout(task_emb)
        task_emb = task_emb.repeat_interleave(x.shape[0] // tasks.shape[0], 0)  # [n_q, n_task, d]
        x = torch.cat((task_emb, x), dim=1)
        z = x.view(-1, x.shape[-1])
        edge_class = torch.tensor([0, 1, 2]).to(self.device)
        input_emb = self.GNN(z, edge_index, edge_class, edge_type.long(), edge_w)
        input_emb = input_emb.contiguous().view(tasks.shape[0], x.shape[0] // tasks.shape[0], x.shape[1], x.shape[2])

        return input_emb

    def forward_subgraph_embedding(self, x, edge_index, edge_w, edge_type, tasks):
        input_emb = self.forward_embedding(x, edge_index, edge_w, edge_type, tasks)
        input_emb = input_emb.mean(1)
        graph_emb = input_emb[:, 0] + input_emb[:, 1:].mean(1)
        return graph_emb

    def forward_inductive(self, x, edge_index, edge_w, edge_type, tasks, tgt_s_idx, tgt_q_idx, n_s, n_q):
        """
        :param sample_emb: [n_q, n_s+1, d]
        :param task_id: [n_task]
        :param support_y: [n_s, n_task]
        :param query_y: [n_q, n_task]
        :param return_auxi
        :return: support_logit: [n_q, n_s]
                 query_logit: [n_q, 1]
                 tgt_s_y: [n_q, n_s]
                 tgt_q_y: [n_q, 1]
                 graph_emb: [d]
        """
        input_emb = self.forward_embedding(x, edge_index, edge_w, edge_type, tasks)

        # tgt_s_idx: [n_s,2], tgt_q_idx: [1,2]
        if self.rel_norm:
            support_sample = torch.cat([F.normalize(input_emb[:, :, tgt_s_idx[0, :], :], p=2, dim=-1),
                                        F.normalize(input_emb[:, :, tgt_s_idx[1, :], :], p=2, dim=-1)], dim=-1)
            query_sample = torch.cat([F.normalize(input_emb[:, :, tgt_q_idx[0, :], :], p=2, dim=-1),
                                      F.normalize(input_emb[:, :, tgt_q_idx[1, :], :], p=2, dim=-1)], dim=-1)
        else:
            support_sample = torch.cat([input_emb[:, :, tgt_s_idx[0, :], :], input_emb[:, :, tgt_s_idx[1, :], :]],
                                       dim=-1)  # [n_q, n_s, d*2]
            query_sample = torch.cat([input_emb[:, :, tgt_q_idx[0, :], :], input_emb[:, :, tgt_q_idx[1, :], :]],
                                     dim=-1)  # [n_q, 1, d*2]
        support_logit = self.classifier(support_sample.contiguous().view(n_q * n_s, -1))

        query_logit = self.classifier(query_sample.contiguous().view(n_q, -1))  # [n_q, 1]
        input_emb = input_emb.mean(1)
        graph_emb = input_emb[:, 0] + input_emb[:, 1:].mean(1)
        return support_logit, query_logit, graph_emb