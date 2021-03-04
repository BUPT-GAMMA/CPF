import optuna
import torch
import torch.optim as optim
import dgl
import numpy as np

from distill_dgl import model_train, choose_model, distill_test
from data.get_cascades import load_cascades
from data.utils import load_tensor_data, initialize_label, set_random_seed, choose_path


class AutoML(object):
    """
    Args:
        func_search: function to obtain hyper-parameters to search
    """

    def __init__(self, kwargs, func_search):
        self.default_params = kwargs
        self.dataset = kwargs['dataset']
        self.seed = kwargs['seed']
        self.func_search = func_search
        self.n_trials = kwargs['ntrials']
        self.n_jobs = kwargs['njobs']
        # self.model = None
        self.best_results = None
        self.preds = None
        self.labels = None
        self.output = None

    def _objective(self, trials):
        params = self.default_params
        params.update(self.func_search(trials))
        results, self.preds, self.labels, self.output = raw_experiment(params)
        if self.best_results is None or results['ValAcc'] > self.best_results['ValAcc']:
            self.best_results = results
        return results['ValAcc']

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        # self.best_value = self.best_value.item()
        return self.best_results, self.preds, self.labels, self.output
        # return self.best_value, self.model


def raw_experiment(configs):
    output_dir, cascade_dir = choose_path(configs)
    # random seed
    set_random_seed(configs['seed'])
    # load_data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
        load_tensor_data(configs['model_name'], configs['dataset'], configs['labelrate'], configs['device'])
    labels_init = initialize_label(idx_train, labels_one_hot).to(configs['device'])
    idx_no_train = torch.LongTensor(
        np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(configs['device'])
    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(configs['device'])
    byte_idx_train[idx_train] = True
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(configs['device'])
    G.ndata['feat'] = features
    G.ndata['feat'].requires_grad_()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print('Loading cascades...')
    cas = load_cascades(cascade_dir, configs['device'], final=True)

    model = choose_model(configs, G, G.ndata['feat'], labels, byte_idx_train, labels_one_hot)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['lr'],
                           weight_decay=configs['wd'])
    acc_val = model_train(configs, model, optimizer, G, labels_init,
                          labels, idx_no_train, idx_train, idx_val, idx_test, cas)
    acc_test, logp, same_predict = distill_test(configs, model, G, labels_init, labels, idx_test, cas)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    results = dict(TestAcc=acc_test.item(), ValAcc=acc_val.item(), SamePredict=same_predict)
    return results, preds, labels, output
