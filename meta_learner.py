
import time
import torch
import logging
import random
from tqdm import tqdm
from copy import deepcopy
from torch import nn
import numpy as np
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from models import MAML, NCESoftmaxLoss
from models.model import KRGTS
from models.selector import Auxiliary_Selector, TaskSelector_graph
from dataset import FewshotMolDataset, dataset_sampler
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()

class MovingAVG:
    def __init__(self):
        self.count = 0
        self.avg = 0

    def get_avg(self):
        return self.avg

    def update(self, x):
        self.count += 1
        self.avg = self.avg + (x - self.avg) / self.count


class MetaLearner:
    def __init__(self, args, device):
        self.batch_size = args.task_batch_size
        self.args = args
        self.device = device
        self.n_support, self.n_query = args.n_support, args.n_query
        self.dataset = FewshotMolDataset(root=args.data_root, name=args.dataset, device=self.device,
                                         workers=args.workers, chunk_size=args.chunk_size)
        self.train_task_range, self.test_task_range = (torch.tensor(self.dataset.train_task_range).to(self.device),
                                                       torch.tensor(self.dataset.test_task_range).to(self.device))

        # filter tasks whose support molecules are less than settings
        self.train_task_range = self.train_task_range[
            torch.where((~torch.isnan(self.dataset.data.y[:, self.train_task_range])).sum(
                0) > 2 * self.n_support + self.n_query, True, False)].cpu().tolist()
        self.test_task_range = self.test_task_range[
            torch.where((~torch.isnan(self.dataset.data.y[:, self.test_task_range])).sum(0) > 2*self.n_support, True,
                        False)].cpu().tolist()

        # model initiation
        model = KRGTS(args=args, task_num=self.dataset.total_tasks,
                      train_task_num=self.dataset.n_task_train, device=device)
        self.maml = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.opt = optim.AdamW(self.maml.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.inner_update_step = args.inner_update_step

        # set the number of auxiliary tasks
        if self.args.train_auxi_task_num is None:
            self.train_auxi_task_num = len(self.train_task_range) - 1
        else:
            self.train_auxi_task_num = min(args.train_auxi_task_num, len(self.train_task_range) - 1)
        if self.args.test_auxi_task_num is None:
            self.test_auxi_task_num = len(self.train_task_range)
        else:
            self.test_auxi_task_num = min(args.test_auxi_task_num, len(self.train_task_range))

        # meta training tasks sampler initiation
        self.task_selector = TaskSelector_graph(input_size=args.rel_hidden_dim).to(self.device)
        self.task_opt = optim.AdamW(self.task_selector.parameters(), lr=args.task_lr, weight_decay=args.weight_decay)
        self.task_reward_avg = MovingAVG()

        # auxiliary task sampler initiation
        self.auxiliary_selector = Auxiliary_Selector(input_size=args.rel_hidden_dim,
                                                     task_range=self.train_task_range).to(self.device)
        self.auxiliary_opt = optim.AdamW(self.auxiliary_selector.parameters(), lr=args.auxi_lr,
                                         weight_decay=args.weight_decay)
        self.gamma = args.auxi_gamma

        # loss settings
        self.nce_loss = NCESoftmaxLoss(t=args.nce_t)
        self.args.pool_num = min(self.args.pool_num, len(self.train_task_range))
        self.finger_vec = torch.stack([data.fingerprint for data in self.dataset])
        self.group_vec = torch.stack([data.groupprint for data in self.dataset])

    def update_inner(self, s_data, q_data, task_id, sca_adj, gro_adj):
        rewards = []
        returns = []
        probs = []
        model = self.maml.clone()
        model.train()

        mask_task = ((~torch.isnan(s_data.y[:, self.train_task_range])).sum(0).bool() | (~torch.isnan(
            q_data.y[:, self.train_task_range])).sum(0).bool())
        mask_task[self.train_task_range.index(task_id)] = False

        # sample auxiliary tasks
        with torch.no_grad():
            sampled_task = torch.tensor([task_id]).to(self.device)
            s_logit, q_logit, s_label, q_label, graph_f = model(s_data, q_data, sca_adj, gro_adj,
                                                                sampled_task.unsqueeze(0))
            graph_emb = self.task_emb_get(model, task_id, mask_task)
        auxi_tasks, auxi_prob = self.sample_auxiliary(graph_emb, self.train_auxi_task_num, mask_task)
        probs.append(torch.log(auxi_prob).sum())
        auxi_tasks = [self.train_task_range[ind] for ind in auxi_tasks]
        sampled_task = torch.tensor([task_id] + auxi_tasks).to(self.device)

        # adapt
        for _ in range(self.args.inner_update_step):
            s_logit, q_logit, s_label, q_label, graph_f = model(s_data, q_data, sca_adj, gro_adj,
                                                                sampled_task.unsqueeze(0))
            inner_loss = self.cls_criterion(s_logit, s_label)
            model.adapt(inner_loss)
        rewards.append(-self.cls_criterion(q_logit, q_label).detach())

        # sample auxiliary tasks
        with torch.no_grad():
            graph_emb = self.task_emb_get(model, task_id, mask_task)
        auxi_tasks, auxi_prob = self.sample_auxiliary(graph_emb, self.train_auxi_task_num, mask_task)
        probs.append(torch.log(auxi_prob).sum())
        auxi_tasks = [self.train_task_range[ind] for ind in auxi_tasks]
        sampled_task = torch.tensor([task_id] + auxi_tasks).to(self.device)

        # test
        s_logit, q_logit, s_label, q_label, graph_f = model(s_data, q_data, sca_adj, gro_adj, sampled_task.unsqueeze(0))
        eval_loss = self.cls_criterion(q_logit, q_label)
        rewards.append(-eval_loss.detach())
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return eval_loss, graph_f, returns, probs

    def train_step(self, epoch, logger_path):

        selected_ids, selected_tasks, selected_prob = self.sample_tasks(epoch)
        eval_losses = []
        graph_f1s, graph_f2s = [], []
        returns = None
        probs = None

        for task_id, (s_data1, q_data1, sca_adj1, gro_adj1, s_data2, q_data2, sca_adj2,
                      gro_adj2) in zip(selected_ids, selected_tasks):

            eval_loss1, graph_f1, returns1, probs1 = self.update_inner(s_data1, q_data1, task_id, sca_adj1, gro_adj1)
            eval_loss2, graph_f2, returns2, probs2 = self.update_inner(s_data2, q_data2, task_id, sca_adj2, gro_adj2)
            eval_losses += [eval_loss1, eval_loss2]
            graph_f1s.append(graph_f1)
            graph_f2s.append(graph_f2)

            # collect the returns and probs
            if returns is None:
                returns = returns1
                returns.extend(returns2)
                probs = probs1
                probs.extend(probs2)
            else:
                returns.extend(returns1)
                returns.extend(returns2)
                probs.extend(probs1)
                probs.extend(probs2)

        # update meta learner
        tgt_f1, tgt_f2 = torch.vstack(graph_f1s), torch.vstack(graph_f2s)
        loss_contr = self.nce_loss(tgt_f1, tgt_f2)
        loss_cls = torch.stack(eval_losses).mean()
        self.opt.zero_grad()
        loss = loss_cls + loss_contr * self.args.contr_w
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), 1)
        self.opt.step()

        # update task selector:
        loss_task = -torch.log(selected_prob).sum()
        reward = loss_contr.item()
        loss_task *= (reward - self.task_reward_avg.get_avg())
        self.task_reward_avg.update(reward)
        self.task_opt.zero_grad()
        loss_task.backward()
        torch.nn.utils.clip_grad_norm_(self.task_selector.parameters(), 1)
        self.task_opt.step()

        # update auxi selector:
        returns = torch.stack(returns)
        probs = torch.stack(probs)
        if self.args.auxi_norm:
            returns_norm = (returns - returns.mean()) / (returns.std(-1) + 1e-9)
        else:
            returns_norm = returns
        loss_auxi = -(returns_norm * probs).mean()

        self.auxiliary_opt.zero_grad()
        loss_auxi.backward()
        torch.nn.utils.clip_grad_norm_(self.auxiliary_selector.parameters(), 1)
        self.auxiliary_opt.step()

        return loss_cls.item(), reward, returns.mean()

    def test_step(self, logger_path, test_auxi_task_num=None):
        auc_scores = []
        for task_i in tqdm(self.test_task_range, desc='eval'):
            # sample data
            s_data, q_data, s_data_id, q_data_id = dataset_sampler(self.dataset, self.n_support, self.n_query,
                                                                   tgt_id=task_i, inductive=True)
            s_data_id = torch.LongTensor(s_data_id).to(self.device)
            s_data = Batch.from_data_list(s_data).to(self.device)
            test_auxi_task_num = self.test_auxi_task_num if test_auxi_task_num is None else test_auxi_task_num

            # initialize model
            model = self.maml.clone()
            model.train()

            # sample auxiliary tasks
            idx = torch.randint(len(q_data), size=(16,), requires_grad=False, device=self.device)
            q_data_ = Batch.from_data_list(q_data[idx]).to(self.device)
            mask_task = ((~torch.isnan(s_data.y[:, self.train_task_range])).sum(0).bool() | (~torch.isnan(
                q_data_.y[:, self.train_task_range])).sum(0).bool())
            with torch.no_grad():
                train_graph_emb = self.task_emb_get(model, task_i, mask_task)
                auxi_tasks, auxi_prob = self.sample_auxiliary(train_graph_emb, test_auxi_task_num, mask_task)
                auxi_tasks = [self.train_task_range[ind] for ind in auxi_tasks]

            # inner update
            adapt_q_iter = iter(DataLoader(q_data, batch_size=self.args.test_batch_size, shuffle=True))
            for _ in range(self.args.inner_update_step):
                adapt_q_data = next(adapt_q_iter)
                sca_adj, gro_adj = self.compute_similarity_chunk(s_data_id, adapt_q_data.id, self.args.rel_top_k)
                adapt_q_data = adapt_q_data.to(self.device)
                sampled_task = torch.tensor([task_i] + auxi_tasks).to(self.device)
                s_logit, q_logit, s_label, q_label, _ = model(s_data, adapt_q_data, sca_adj, gro_adj,
                                                              sampled_task.unsqueeze(0))
                inner_loss = self.cls_criterion(s_logit, s_label)
                model.adapt(inner_loss)

            # eval query data
            model.eval()
            with torch.no_grad():
                train_graph_emb = self.task_emb_get(model, task_i, mask_task)
                auxi_tasks, auxi_prob = self.sample_auxiliary(train_graph_emb, test_auxi_task_num, mask_task)
                auxi_tasks = [self.train_task_range[ind] for ind in auxi_tasks]
            y_pred, y_true = [], []
            query_loader = DataLoader(q_data, batch_size=self.args.test_batch_size, shuffle=False)
            with torch.no_grad():
                for iter_q_data in query_loader:
                    sca_adj, gro_adj = self.compute_similarity_chunk(s_data_id, iter_q_data.id, self.args.rel_top_k)
                    iter_q_data = iter_q_data.to(self.device)
                    sampled_task = torch.tensor([task_i] + auxi_tasks).to(self.device)
                    s_logit, q_logit, s_label, q_label, _ = model(s_data, iter_q_data, sca_adj, gro_adj,
                                                                  sampled_task.unsqueeze(0))
                    q_logit = torch.sigmoid(q_logit).cpu().view(-1)
                    q_label = q_label.cpu().view(-1)
                    y_pred.append(q_logit)
                    y_true.append(q_label)

                y_true = torch.cat(y_true, dim=0).numpy()
                y_pred = torch.cat(y_pred, dim=0).numpy()
                score = roc_auc_score(y_true, y_pred)
                auc_scores.append(score)
        return np.mean(auc_scores)

    def compute_similarity_chunk(self, s_datas, q_datas, top_k):
        s_fingerprint = self.finger_vec[s_datas]
        q_fingerprint = self.finger_vec[q_datas]
        s_groupprint = self.group_vec[s_datas]
        q_groupprint = self.group_vec[q_datas]

        sca_vec = torch.cat((s_fingerprint.unsqueeze(-3).repeat_interleave(repeats=q_datas.shape[-1], dim=-3),
                             q_fingerprint.unsqueeze(-2)), dim=-2)
        sca_vec = sca_vec.view(-1, sca_vec.shape[-2], sca_vec.shape[-1])
        sca_vec_ = 1 - sca_vec
        sca_sim = sca_vec.bmm(sca_vec.permute(0, 2, 1)) / (2214 - sca_vec_.bmm(sca_vec_.permute(0, 2, 1)) + 1e-9)
        torch.diagonal(sca_sim[:, :, :], dim1=-2, dim2=-1).zero_()

        gro_vec = torch.cat((s_groupprint.unsqueeze(-3).repeat_interleave(q_datas.shape[-1], dim=-3),
                             q_groupprint.unsqueeze(-2)), dim=-2)
        gro_vec = gro_vec.view(-1, gro_vec.shape[-2], gro_vec.shape[-1])
        gro_vec_ = 1 - gro_vec
        gro_sim = gro_vec.bmm(gro_vec.permute(0, 2, 1)) / (49 - gro_vec_.bmm(gro_vec_.permute(0, 2, 1)) + 1e-9)
        torch.diagonal(gro_sim[:, :, :], dim1=-2, dim2=-1).zero_()
        if top_k > 0:
            _, topk_indices_gro = torch.topk(gro_sim * (-1), s_datas.shape[-1] + 1 - top_k, dim=-1)
            gro_sim.scatter_(-1, topk_indices_gro, 0)
            _, topk_indices_sca = torch.topk(sca_sim * (-1), s_datas.shape[-1] + 1 - top_k, dim=-1)
            sca_sim.scatter_(-1, topk_indices_sca, 0)
        return sca_sim, gro_sim

    def sample_datas(self, tgt_ids):
        batch_s_id1, batch_q_id1, batch_s_id2, batch_q_id2 = [], [], [], []
        for idx, task_id in enumerate(tgt_ids):
            s_data1, q_data1, s_id1, q_id1 = dataset_sampler(self.dataset, self.n_support, self.n_query, task_id)
            s_data2, q_data2, s_id2, q_id2 = dataset_sampler(self.dataset, self.n_support, self.n_query, task_id)
            if idx == 0:
                batch_s_data1 = s_data1
                batch_q_data1 = q_data1
                batch_s_data2 = s_data2
                batch_q_data2 = q_data2
            else:
                batch_s_data1 += s_data1
                batch_q_data1 += q_data1
                batch_s_data2 += s_data2
                batch_q_data2 += q_data2
            batch_s_id1.append(s_id1)
            batch_q_id1.append(q_id1)
            batch_s_id2.append(s_id2)
            batch_q_id2.append(q_id2)
        batch_s_data1 = Batch.from_data_list(batch_s_data1)
        batch_q_data1 = Batch.from_data_list(batch_q_data1)
        batch_s_data2 = Batch.from_data_list(batch_s_data2)
        batch_q_data2 = Batch.from_data_list(batch_q_data2)
        batch_s_id1 = torch.LongTensor(batch_s_id1).to(self.device)
        batch_q_id1 = torch.LongTensor(batch_q_id1).to(self.device)
        batch_s_id2 = torch.LongTensor(batch_s_id2).to(self.device)
        batch_q_id2 = torch.LongTensor(batch_q_id2).to(self.device)
        batch_sca_adj1, batch_gro_adj1 = self.compute_similarity_chunk(batch_s_id1, batch_q_id1, self.args.rel_top_k)
        batch_sca_adj2, batch_gro_adj2 = self.compute_similarity_chunk(batch_s_id2, batch_q_id2, self.args.rel_top_k)
        return (batch_s_data1, batch_q_data1, batch_s_data2, batch_q_data2, batch_sca_adj1,
                batch_gro_adj1, batch_sca_adj2, batch_gro_adj2)

    def sample_data(self, tgt_ids):
        batch_s_id1, batch_q_id1 = [], []
        for idx, task_id in enumerate(tgt_ids):
            s_data1, q_data1, s_id1, q_id1 = dataset_sampler(self.dataset, self.n_support, self.n_query, task_id)
            if idx == 0:
                batch_s_data1 = s_data1
                batch_q_data1 = q_data1
            else:
                batch_s_data1 += s_data1
                batch_q_data1 += q_data1
            batch_s_id1.append(s_id1)
            batch_q_id1.append(q_id1)
        batch_s_data1 = Batch.from_data_list(batch_s_data1)
        batch_q_data1 = Batch.from_data_list(batch_q_data1)
        batch_s_id1 = torch.LongTensor(batch_s_id1).to(self.device)
        batch_q_id1 = torch.LongTensor(batch_q_id1).to(self.device)
        batch_sca_adj1, batch_gro_adj1 = self.compute_similarity_chunk(batch_s_id1, batch_q_id1, self.args.rel_top_k)
        return batch_s_data1, batch_q_data1, batch_sca_adj1, batch_gro_adj1


    def sample_tasks(self, epoch):
        model = self.maml.clone()
        model.eval()
        with torch.no_grad():
            tasks_pool = []
            train_tasks = torch.LongTensor(self.train_task_range).to(self.device)
            rand_tasks_pool_ids = torch.randperm(len(train_tasks))
            task_pool_ids = train_tasks[rand_tasks_pool_ids[0:self.args.pool_num]]
            batch_s1, batch_q1, batch_s2, batch_q2, batch_sca1, batch_gro1, batch_sca2, batch_gro2 = self.sample_datas(
                task_pool_ids)
            graph_emb1 = model.get_embedding(batch_s1, batch_q1, batch_sca1, batch_gro1, task_pool_ids.view(-1, 1))
            graph_emb2 = model.get_embedding(batch_s2, batch_q2, batch_sca2, batch_gro2, task_pool_ids.view(-1, 1))
            graph_emb = torch.cat((graph_emb1, graph_emb2), dim=-1).view(self.args.pool_num * 2,
                                                                        self.args.rel_hidden_dim)

            for j in range(self.args.pool_num):
                st_s_id = j * self.n_support * 2
                end_s_id = (j + 1) * self.n_support * 2
                st_q_id = j * self.n_query
                end_q_id = (j + 1) * self.n_query
                batch_s_data_1 = Batch.from_data_list(batch_s1[st_s_id:end_s_id]).to(self.device)
                batch_q_data_1 = Batch.from_data_list(batch_q1[st_q_id:end_q_id]).to(self.device)
                batch_s_data_2 = Batch.from_data_list(batch_s2[st_s_id:end_s_id]).to(self.device)
                batch_q_data_2 = Batch.from_data_list(batch_q2[st_q_id:end_q_id]).to(self.device)
                tasks_pool.append((batch_s_data_1, batch_q_data_1, batch_sca1[st_q_id:end_q_id, :, :],
                                   batch_gro1[st_q_id:end_q_id, :, :], batch_s_data_2, batch_q_data_2,
                                   batch_sca2[st_q_id:end_q_id, :, :], batch_gro2[st_q_id:end_q_id, :, :]))
        w = self.task_selector(graph_emb)  # [n_pool*2]
        w = w.reshape(-1, 2).mean(-1)  # [n_pool]
        selected_index = self.task_selector.sample(w.cpu().tolist(), self.args.inner_tasks // 2)
        selected_prob = w[selected_index]
        selected_tasks, selected_ids = [], []
        for idx in selected_index:
            selected_tasks.append(tasks_pool[idx])
            selected_ids.append(task_pool_ids[idx])
        return selected_ids, selected_tasks, selected_prob

    def task_emb_get(self, model, t_id, mask_task):
        tasks = self.train_task_range.copy()
        tasks = torch.tensor(tasks).to(self.device)[mask_task].tolist()
        tasks.extend([t_id])
        batch_s, batch_q, batch_sca, batch_gro = self.sample_data(tasks)
        tasks = torch.LongTensor(tasks).to(self.device)
        graph_emb = model.get_embedding(batch_s, batch_q, batch_sca, batch_gro, tasks.unsqueeze(1))
        return graph_emb

    def sample_auxiliary(self, graph_emb, auxi_task_num, mask_task):
        probs = torch.zeros(len(self.train_task_range)).to(self.device)
        mask = ~mask_task
        probs[~mask] = torch.concat(
            [self.auxiliary_selector(graph_emb[-1], graph_emb[0:-1][num:num + 300]) if (num + 300) < len(
                graph_emb) else self.auxiliary_selector(graph_emb[-1], graph_emb[0:-1][num:]) for num in
             range(0, len(graph_emb), 300)])
        probs[mask] = -1e9
        probs = torch.softmax(probs, dim=-1)
        if auxi_task_num > mask_task.sum():
            auxi_task_num = mask_task.sum() - 1
        if auxi_task_num <= 0:
            auxi_task_num = 1
        selected_ids = torch.topk(probs, auxi_task_num)[1].tolist()
        selected_prob = probs[selected_ids]
        return selected_ids, selected_prob

