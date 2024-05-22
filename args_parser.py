import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="Log/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--random_seed", default=0, type=int)

    # dataset
    parser.add_argument("--data_root", default='./data/', type=str)
    parser.add_argument("--dataset", default='tox21', type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--chunk_size", default=5000, type=int)

    # mol encoder
    parser.add_argument("--mol_num_layer", default=5, type=int)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--JK", default='last', type=str)
    parser.add_argument("--mol_dropout", default=0.1, type=float)
    parser.add_argument("--mol_graph_pooling", default='mean', type=str)
    parser.add_argument("--mol_gnn_type", default='gin', type=str)
    parser.add_argument("--mol_batch_norm", default=1, type=int)
    parser.add_argument("--mol_pretrain_load_path", default=None)

    # relation net
    parser.add_argument("--rel_layer", default=2, type=int)
    parser.add_argument("--rel_top_k", default=None, type=int)
    parser.add_argument("--rel_dropout", default=0.1, type=float)
    parser.add_argument("--rel_pre_dropout", default=0.1, type=float)
    parser.add_argument("--rel_nan_type", default='nan', type=str, choices=['nan', '0', '1'])
    parser.add_argument("--rel_hidden_dim", default=128, type=int)
    parser.add_argument("--rel_batch_norm", default=1, type=int)
    parser.add_argument("--rel_norm", default=1, type=int)
    parser.add_argument("--rel_nan_w", default=2, type=float)  # 0 or 2
    parser.add_argument("--num_relation_attr", default=3, type=int)

    # maml
    parser.add_argument("--inner_lr", default=5e-1, type=float)
    parser.add_argument("--meta_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int)
    parser.add_argument("--inner_update_step", default=1, type=int)
    parser.add_argument("--inner_tasks", default=10, type=int)

    # few-shot
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--n_support", default=10, type=int)
    parser.add_argument("--n_query", default=16, type=int)
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--train_auxi_task_num", default=None, type=int)
    parser.add_argument("--test_auxi_task_num", default=None, type=int)
    parser.add_argument("--task_batch_size", default=256, type=int)

    # contrastive
    parser.add_argument("--nce_t", default=0.08, type=float)
    parser.add_argument("--contr_w", default=0.05, type=float)

    # meta training selector
    parser.add_argument("--pool_num", default=10, type=int)
    parser.add_argument("--task_lr", default=5e-4, type=float)

    # auxiliary selector
    parser.add_argument("--auxi_lr", default=5e-4, type=float)
    parser.add_argument("--auxi_norm", default=0, type=int)
    parser.add_argument("--s_weight", default=0.3, type=float)
    parser.add_argument("--q_weight", default=0.7, type=float)
    parser.add_argument("--auxi_gamma", default=0.95, type=float)

    args = parser.parse_args()


    if args.rel_nan_w != 0:
        args.num_relation_attr = 4
    if args.rel_top_k is None:
        args.rel_top_k = args.n_support - 1 if args.n_support > 1 else 1

    return args
