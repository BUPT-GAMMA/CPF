import numpy as np
import scipy.sparse as sp
import torch
import os
import time
from pathlib import Path
from data.get_dataset import load_dataset_and_split


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


def normalize_features(features):
    features = normalize(features)
    return features


def initialize_label(idx_train, labels_one_hot):
    labels_init = torch.ones_like(labels_one_hot) / len(labels_one_hot[0])
    labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init


def split_double_test(dataset, idx_test):
    test_num = len(idx_test)
    idx_test1 = idx_test[:int(test_num/2)]
    idx_test2 = idx_test[int(test_num/2):]
    return idx_test1, idx_test2


def preprocess_adj(model_name, adj):
    return normalize_adj(adj)


def preprocess_features(model_name, features):
    return features


def load_ogb_data(dataset, device):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name="ogbn-"+dataset, root='data')
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    labels = labels.squeeze()
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()
    features = graph.ndata['feat']
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    return graph, features, labels, idx_train, idx_val, idx_test


def load_tensor_data(model_name, dataset, labelrate, device):
    if dataset in ['composite', 'composite2', 'composite3']:
        adj, features, labels_one_hot, idx_train, idx_val, idx_test = load_composite_data(dataset)
    else:
        # config_file = os.path.abspath('data/dataset.conf.yaml')
        adj, features, labels_one_hot, idx_train, idx_val, idx_test = load_dataset_and_split(labelrate, dataset)
    adj = preprocess_adj(model_name, adj)
    features = preprocess_features(model_name, features)
    adj_sp = adj.tocoo()
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = labels_one_hot.argmax(axis=1)
    # labels, labels_init = initialize_label(idx_train, labels_one_hot)
    labels = torch.LongTensor(labels)
    labels_one_hot = torch.FloatTensor(labels_one_hot)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # adj_sp = sp.coo_matrix(adj.numpy(), dtype=float)
    print('Device: ', device)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    labels_one_hot = labels_one_hot.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    return adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test


def load_composite_data(dataset):
    base_dir = Path.cwd().joinpath('data', dataset)
    adj = np.loadtxt(str(base_dir.joinpath('adj')))
    features = np.loadtxt(str(base_dir.joinpath('features')))
    labels_one_hot = np.loadtxt(str(base_dir.joinpath('labels')))
    idx_train = np.loadtxt(str(base_dir.joinpath('idx_train')))
    idx_val = np.loadtxt(str(base_dir.joinpath('idx_val')))
    idx_test = np.loadtxt(str(base_dir.joinpath('idx_test')))
    adj = sp.csr_matrix(adj)
    # adj = normalize_adj(adj)
    features = sp.csr_matrix(features)
    # features = normalize_features(features)
    # labels, labels_init = initialize_label(idx_train, labels_one_hot)

    return adj, features, labels_one_hot, idx_train, idx_val, idx_test


def table_to_dict(adj):
    adj = adj.cpu().numpy()
    # print(adj)
    # adj = adj.todense()
    adj_list = dict()
    for i in range(len(adj)):
        adj_list[i] = set(np.argwhere(adj[i] > 0).ravel())
    return adj_list


def matrix_pow(m1, n, m2):
    t = time.time()
    m1 = sp.csr_matrix(m1)
    m2 = sp.csr_matrix(m2)
    ans = m1.dot(m2)
    for i in range(n-2):
        ans = m1.dot(ans)
    ans = torch.FloatTensor(ans.todense())
    print(time.time() - t)
    return ans


def quick_matrix_pow(m, n):
    t = time.time()
    E = torch.eye(len(m))
    while n:
        if n % 2 != 0:
            E = torch.matmul(E, m)
        m = torch.matmul(m, m)
        n >>= 1
    print(time.time() - t)
    return E


def row_normalize(data):
    return (data.t() / torch.sum(data.t(), dim=0)).t()


def np_normalize(matrix):
    from sklearn.preprocessing import normalize
    """Normalize the matrix so that the rows sum up to 1."""
    matrix[np.isnan(matrix)] = 0
    return normalize(matrix, norm='l1', axis=1)


def check_writable(dir, overwrite=True):
    import shutil
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite:
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        pass


def check_readable(dir):
    if not os.path.exists(dir):
        print(dir)
        raise ValueError(f'No such a directory or file!')


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_path(conf):
    if 'assistant' not in conf.keys():
        teacher_str = conf['teacher']
    elif conf['assistant'] == 0:
        teacher_str = 'nasty_' + conf['teacher']
    elif conf['assistant'] == 1:
        teacher_str = 'reborn_' + conf['teacher']
    else:
        raise ValueError(r'No such assistant')
    if conf['student'] == 'PLP' and conf['ptype'] == 0:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']) + '_ind')
    elif conf['student'] == 'PLP' and conf['ptype'] == 1:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']) + '_tra')
    else:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']))
    check_writable(output_dir, overwrite=False)
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str,
                                      'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']), 'cascade')
    # check_readable(cascade_dir)
    return output_dir, cascade_dir

