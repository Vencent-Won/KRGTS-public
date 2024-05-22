
import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
import numpy as np


class TaskSelector_graph(nn.Module):
    def __init__(self, input_size):
        super(TaskSelector_graph, self).__init__()
        self.z1 = nn.Sequential(nn.Linear(input_size, 1), nn.Tanh())
        self.z2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        final_adj = torch.matmul(x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-9),
                                 (x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-9)).t())
        A_norm = final_adj - 1e9 * torch.less_equal(final_adj, 0.8)
        A_norm = A_norm.softmax(-1)
        output = self.dropout(F.leaky_relu(self.z2(A_norm.mm(x))))
        output = self.z1(output)
        output = output.view(-1)
        prob = torch.softmax(output, dim=0)

        return prob

    def sample(self, prob, n):
        # prob: List
        prob = np.array(prob)
        prob /= prob.sum()
        res = np.random.choice(len(prob), n, replace=False, p=prob).tolist()

        return res

class Auxiliary_Selector(nn.Module):
    def __init__(self, input_size, task_range):
        super(Auxiliary_Selector, self).__init__()
        self.z1 = nn.Sequential(nn.Linear(input_size*2, int(input_size)), nn.ReLU(),
                                nn.Linear(int(input_size), int(input_size)), nn.ReLU(),
                                nn.Linear(int(input_size), 1), nn.Tanh())
        self.task_range = task_range
        for j in range(len(self.z1)):
            module = self.z1[j]
            if isinstance(module, nn.Linear):
                if j == len(self.z1) - 1:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.constant_(module.bias, 0)
                else:
                    nn.init.orthogonal_(module.weight, gain=1)
                    nn.init.constant_(module.bias, 0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x):

        output = torch.cat([x1.repeat(x.shape[0], 1), x], dim=-1)

        return self.z1(output).view(-1)


    def sample(self, prob, task_mask, n):
        # prob: List
        prob = np.array(prob)
        for i, m in enumerate(task_mask):
            if m == 1:
                prob[i] = 0
        prob = np.array(prob) / np.sum(prob)
        try:
            if n > np.count_nonzero(prob):
                sample_id = np.random.choice(len(prob), np.count_nonzero(prob), replace=False, p=prob).tolist()
            else:
                sample_id = np.random.choice(len(prob), n, replace=False, p=prob).tolist()
        except:
            sample_id = np.random.choice(len(prob), n, replace=False).tolist()

        return sample_id
