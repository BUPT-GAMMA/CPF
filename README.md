# CPF

The official code of WWW2021 paper: Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-fixed-10-node-per)](https://paperswithcode.com/sota/node-classification-on-cora-fixed-10-node-per)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-fixed-5-node-per)](https://paperswithcode.com/sota/node-classification-on-cora-fixed-5-node-per)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-3)](https://paperswithcode.com/sota/node-classification-on-cora-3)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-1)](https://paperswithcode.com/sota/node-classification-on-cora-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-05)](https://paperswithcode.com/sota/node-classification-on-cora-05)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-amz-computers)](https://paperswithcode.com/sota/node-classification-on-amz-computers?p=extract-the-knowledge-of-graph-neural)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-amz-photo)](https://paperswithcode.com/sota/node-classification-on-amz-photo?p=extract-the-knowledge-of-graph-neural)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=extract-the-knowledge-of-graph-neural)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=extract-the-knowledge-of-graph-neural)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=extract-the-knowledge-of-graph-neural)

## Getting Started

### Requirements

- Python version >= 3.6
- PyTorch version >= 1.7.1
- DGL
- Optuna (optional)

## Usage

### Quick start

1. use `python train_dgl.py --dataset=XXX --teacher=XXX`  to run teacher model.
2. use `python spawn_worker.py --dataset=XXX --teacher=XXX ` to run student model, we provide our hyper-parameters setting as reported in our paper, and an AutoML version for hyper-parameters search. (Our code supports Optuna to search best hyper-parameters for knowledge distillation. You can use `--automl`  to run Optuna code.)

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

## Benchmark Rankings

There are results use several models run on different benchmark datasets. Our experiments settings are available in the following form and the `pwc.conf.yaml` file. For simple usage, please try AutoML for hyper-parameters search.

Note: 

- Remember to change the load data function to load_citation when running public split benchmarks. 
- Use original load data function when running AMZ datasets, remember to slice the test sets to corresponding size.

|                          Benchmark                           | Model         | Acc    | layer | emb_dim | feat_drop | attn_drop | lr   | wd   |
| :----------------------------------------------------------: | ------------- | ------ | ----- | ------- | --------- | --------- | ---- | ---- |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-fixed-10-node-per)](https://paperswithcode.com/sota/node-classification-on-cora-fixed-10-node-per) | CPF-tra-GCNII | 84.1%  | 6     | 16      | 0.2       | 0.5       | 5e-3 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-fixed-5-node-per)](https://paperswithcode.com/sota/node-classification-on-cora-fixed-5-node-per) | CPF-tra-APPNP | 80.26% | 8     | 32      | 0.2       | 0.2       | 5e-3 | 5e-4 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-3)](https://paperswithcode.com/sota/node-classification-on-cora-3) | CPF-tra-GCNII | 84.18% | 9     | 8       | 0.5       | 0.8       | 5e-3 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-1)](https://paperswithcode.com/sota/node-classification-on-cora-1) | CPF-ind-APPNP | 80.24% | 8     | 16      | 0.8       | 0.2       | 5e-3 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-05)](https://paperswithcode.com/sota/node-classification-on-cora-05) | CPF-ind-APPNP | 77.3%  | 7     | 32      | 0.8       | 0.2       | 1e-3 | 1e-3 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-amz-computers)](https://paperswithcode.com/sota/node-classification-on-amz-computers?p=extract-the-knowledge-of-graph-neural) | CPF-ind-GAT   | 85.5%  | 8     | 16      | 0.2       | 0.5       | 1e-3 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-amz-photo)](https://paperswithcode.com/sota/node-classification-on-amz-photo?p=extract-the-knowledge-of-graph-neural) | CPF-ind-GAT   | 94.1%  | 9     | 32      | 0.5       | 0.5       | 1e-2 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=extract-the-knowledge-of-graph-neural) | CPF-ind-APPNP | 85.3%  | 10    | 64      | 0.8       | 0.8       | 5e-3 | 5e-4 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=extract-the-knowledge-of-graph-neural) | CPF-ind-APPNP | 74.6%  | 6     | 64      | 0.5       | 0.5       | 5e-3 | 1e-2 |
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extract-the-knowledge-of-graph-neural/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=extract-the-knowledge-of-graph-neural) | CPF-tra-GCNII | 83.2%  | 8     | 16      | 0.8       | 0.8       | 1e-2 | 5e-4 |

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{yang2021extract,
  title={Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework},
  author={Cheng Yang and Jiawei Liu and Chuan Shi},
  booktitle={Proceedings of The Web Conference 2021 (WWW ’21)},
  publisher={ACM},
  year={2021}
}
```

## Contact Us

Please open an issue or contact Liu_Jiawei@bupt.edu.cn with any questions.
