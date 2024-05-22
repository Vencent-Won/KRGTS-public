
import sys
import re
import os
import torch
import logging
from tqdm import tqdm
import time
import numpy as np
from meta_learner import MetaLearner
from args_parser import args_parser
from explight import initialize_exp, set_seed, get_dump_path, describe_model, save_model

import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()


class Runner:
    def __init__(self, args, logger_path, device):
        self.args = args
        self.meta_learner = MetaLearner(args, device)
        describe_model(self.meta_learner.maml.module, logger_path, name='model')
        describe_model(self.meta_learner.task_selector, logger_path, name='task_selector')
        self.logger_path = logger_path
        self.device = device

    def run(self):

        best_score = -1
        pbar = tqdm(range(1, self.args.episode + 1))
        cost_time_ls = []
        returns = []
        auxi_return_best = None
        for epoch in pbar:
            start = time.time()
            loss_cls, meta_reward, auxi_return = self.meta_learner.train_step(epoch, self.logger_path)
            cost_time = time.time() - start
            cost_time_ls.append(cost_time)
            returns.append(auxi_return.item())
            if auxi_return_best is None:
                auxi_return_best = auxi_return
                torch.save(self.meta_learner.auxiliary_selector.state_dict(),
                           '{}/auxiliary_selector_param.pkl'.format(self.logger_path))
            elif auxi_return_best <= auxi_return:
                auxi_return_best = auxi_return
                torch.save(self.meta_learner.auxiliary_selector.state_dict(),
                           '{}/auxiliary_selector_param.pkl'.format(self.logger_path))


            pbar.set_description(f"loss={loss_cls:.4f}, meta_reward={meta_reward:.4f}, auxi_reward:{auxi_return:.4f}")
            if epoch % self.args.eval_step == 0:
                self.meta_learner.auxiliary_selector.load_state_dict(
                    torch.load('{}/auxiliary_selector_param.pkl'.format(self.logger_path)))
                score = self.meta_learner.test_step(self.logger_path)
                if score > best_score:
                    best_score = score
                    model_path = self.logger_path
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    if self.args.mol_pretrain_load_path is not None:
                        save_model(self.meta_learner.maml.module, model_path, model_name='pre_model')
                    else:
                        save_model(self.meta_learner.maml.module, model_path, model_name='model')

                logger.info(f"{epoch} | score: {score:.5f}, best_score: {best_score:.5f}")
        logger.info(f"best score: {best_score:.5f}")
        logger.info(f"time cost: {np.mean(cost_time_ls):.5f}s")

        command = []
        for x in sys.argv[1:]:
            if x.startswith('--'):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match('^[a-zA-Z0-9_]+$', x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = ' '.join(command)
        if self.args.mol_pretrain_load_path is None:
            if self.args.n_support == 1:
                with open('result/{}_1shot.txt'.format(self.args.dataset), 'a') as file_handle:
                    file_handle.write(str(command) + ": " + str(best_score))
                    file_handle.write('\n')
            else:
                with open('result/{}_10shot.txt'.format(self.args.dataset), 'a') as file_handle:
                    file_handle.write(str(command) + ": " + str(best_score))
                    file_handle.write('\n')
        else:
            if self.args.n_support == 1:
                with open('pre_result/{}_1shot.txt'.format(self.args.dataset), 'a') as file_handle:
                    file_handle.write(str(command) + ": " + str(best_score))
                    file_handle.write('\n')
            else:
                with open('pre_result/{}_10shot.txt'.format(self.args.dataset), 'a') as file_handle:
                    file_handle.write(str(command) + ": " + str(best_score))
                    file_handle.write('\n')

def main():
    args = args_parser()
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    set_seed(args.random_seed)
    logger = initialize_exp(args)
    logger_path = get_dump_path(args)

    runner = Runner(args, logger_path, device)
    runner.run()


if __name__ == '__main__':
    main()
