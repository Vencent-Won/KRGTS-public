# Knowledge-enhanced Relation Graph and Task Sampling for Few-shot Molecular Property Prediction

This repository is the implementation of **KRGTS** (Knowledge-enhanced Relation Graph and Task Sampling for Few-shot Molecular Property Prediction).

## Framework

<img src="framework/framework.png" alt="framework" style="zoom: 100%;" />


## Environment
To run the code successfully, the following dependencies need to be installed:
```
python           3.7
torch            1.13.1
torch_geometric  2.3.1
torch_scatter    2.1.1
rdkit            2023.3.2
```

## Implementation

### Datasets
For data used in the experiments, please save the contents in the `data` directory.


### Usage

Under the 10-shot setting:

```sh
python run.py --dataset {dataset} --n_support 10 --train_auxi_task_num {num} --test_auxi_task_num {num}
```

Under the 1-shot setting:

```sh
python run.py --dataset {dataset} --n_support 1 --train_auxi_task_num {num} --test_auxi_task_num {num}
```

For Pre-KRGTS, which is initialized with a pretrained GNN, the running script is:

```sh
python run.py --dataset {dataset} --mol_pretrain_load_path pretrained/supervised_contextpred.pth --train_auxi_task_num {num} --test_auxi_task_num {num}
```


### Cite
```
@article{WANG2025122357,
title = {Knowledge-enhanced Relation Graph and Task Sampling for few-shot molecular property prediction},
journal = {Information Sciences},
volume = {718},
pages = {122357},
year = {2025},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2025.122357},
url = {https://www.sciencedirect.com/science/article/pii/S002002552500489X},
author = {Zeyu Wang and Tianyi Jiang and Yao Lu and Xiaoze Bao and Shanqing Yu and Bin Wei and Qi Xuan and Hong Wang},
}
```
