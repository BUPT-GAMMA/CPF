# CPF

The official code of WWW2021 paper: Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework

## Getting Started

### Requirements

- Python version >= 3.6
- PyTorch version >= 1.7.1
- DGL

## Usage

### Quick start

2. use `python train_dgl.py --dataset=XXX --teacher=XXX`  to run teacher model.
2. use `python spawn_worker.py --dataset=XXX --teacher=XXX ` to run student model, we provide our hyper-parameters setting as reported in our paper, and an AutoML version for hyper-parameters search.

### AutoML

Our code supports Optuna to search best hyper-parameters for knowledge distillation. You can use `hyper.py`  to run Optuna code.

## Add your own datasets

You can add your own datasets to folder `data`, the formats should accord to DGL requirements.

## Add your own models

You can add your own teacher or student model by adding them into folder `models`, and following the format of model run.

## Results

There are some results on GCN teacher model, with different datasets and student varients. More results can be seen in our paper.

| Datasets    | GCN (Teacher) | CPF-ind (Student) | CPF-tra (Student) | improvement |
| ----------- | ------------- | ----------------- | ----------------- | ----------- |
| Cora        | 0.8244        | **0.8576**        | 0.8567            | 4.0%        |
| Citeseer    | 0.7110        | 0.7619            | **0.7652**        | 7.6%        |
| Pubmed      | 0.7804        | 0.8080            | **0.8104**        | 3.8%        |
| A-Computers | 0.8318        | **0.8443**        | **0.8443**        | 1.5%        |
| A-Photo     | 0.9072        | **0.9317**        | 0.9248            | 2.7%        |

## Contact Us

Please open an issue or contact Liu_Jiawei@bupt.edu.cn with any questions.