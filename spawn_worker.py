import argparse
import copy
import itertools
from pathlib import Path
from models.utils import get_training_config, check_device
from data.get_dataset import get_experiment_config
from utils.logger import output_results
from hypersearch import AutoML, raw_experiment
from collections import defaultdict, namedtuple


num_layers = [10, 6, 5, 7, 8, 9]
emb_dim = [64, 32, 16, 8]
feat_drop = [0.8, 0.5, 0.2]
attn_drop = [0.2, 0.5, 0.8]
lr = [0.001, 0.005, 0.01]
wd = [0.01, 0.001, 0.0005]

predefined_configs = {
    'ind': {
        'GCN': {
            'cora': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 8,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.8,
                'beta': 1,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GAT': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 6,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'beta': 1,
                'lr': 0.01,
                'wd': 0.01
            },
        },
        'APPNP': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 5,
                'lr': 0.005,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.8,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.01
            },
        },
        'GraphSAGE': {
            'cora': {
                'num_layers': 9,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 9,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.8,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'SGC': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 10,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GCNII': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 8,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 8,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GLP': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        }
    },
    'tra': {
        'GCN': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 8,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.8,
                'beta': 1,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 5,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GAT': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 10,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.5,
                'beta': 1,
                'lr': 0.01,
                'wd': 0.0005
            },
        },
        'APPNP': {
            'cora': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 5,
                'lr': 0.01,
                'wd': 0.0005
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.005,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.001
            },
        },
        'GraphSAGE': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 10,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 9,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'SGC': {
            'cora': {
                'num_layers': 9,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 6,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.8,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GCNII': {
            'cora': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 8,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 8,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GLP': {
            'cora': {
                'num_layers': 6,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        }
    }
}


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')
    parser.add_argument('--device', type=int, default=3, help='CUDA Device')
    parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')
    parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')
    parser.add_argument('--grad', type=int, default=1, help='output grad or not')

    parser.add_argument('--automl', action='store_true', default=False, help='Automl or not')
    parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--njobs', type=int, default=10, help='Number of jobs')
    return parser.parse_args()


def set_configs(configs):
    configs = dict(configs, **predefined_configs[args.ptype][args.teacher][args.dataset])
    training_configs_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    model_name = configs['student'] if configs['distill'] else configs['teacher']
    training_configs = get_training_config(training_configs_path, model_name)
    configs = dict(configs, **training_configs)
    configs['device'] = check_device(configs)
    data_configs_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    configs['division_seed'] = get_experiment_config(data_configs_path)['seed']
    return configs


def func_search(trial):
    return {
        "num_layers": trial.suggest_int("num_layers", 5, 10),
        "emb_dim": trial.suggest_categorical("emb_dim", [64, 32, 16, 8]),
        "feat_drop": trial.suggest_categorical("feat_drop", [0.8, 0.5, 0.2]),
        "attn_drop": trial.suggest_categorical("attn_drop", [0.8, 0.5, 0.2]),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-3, 1e-2]),
        "weight_decay": trial.suggest_categorical("weight_decay", [5e-4, 1e-3, 1e-2]),
    }


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    print()
    return itertools.starmap(Variant, itertools.product(*items.values()))


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for variant in variants:
        args.dataset, args.model, args.seed = variant
        yield copy.deepcopy(args)


if __name__ == '__main__':
    # load_configs
    args = arg_parse(argparse.ArgumentParser())
    configs = set_configs(args.__dict__)
    print(configs)
    # model_train
    variants = list(gen_variants(dataset=[configs['dataset']],
                                 model=[configs['model_name']],
                                 seed=[configs['seed']]))
    print(variants)
    results_dict = defaultdict(list)
    for variant in variants:
        if args.automl:
            tool = AutoML(kwargs=configs, func_search=func_search)
            results, preds, labels, output = tool.run()
        else:
            results, preds, labels, output = raw_experiment(configs)
        results_dict[variant[:]] = [results]
        tablefmt = configs["tablefmt"] if "tablefmt" in configs else "github"
        print("\nFinal results:\n")

        output_results(results_dict, tablefmt)

    # # save outputs
    # save_output(output_dir, preds, labels, output, acc_test, same_predict, G, idx_train, adj, configs)
