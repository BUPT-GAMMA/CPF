from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path

from data.get_cascades import load_cascades
from data.utils import load_tensor_data, initialize_label, \
    matrix_pow, row_normalize, check_writable, check_readable
from data.get_dataset import get_experiment_config

import dgl
from models.GCN import GCN
from models.GAT import GAT
from models.MoNet import MoNet
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.GCNII import GCNII
from models.utils import get_training_config
from dgl.nn.pytorch.conv import SGConv

from utils.logger import get_logger
from utils.metrics import accuracy, my_loss


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--device', type=int, default=6, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')

    # for plp hyper parameters finding
    parser.add_argument('--num_layers', type=int, default=10, help='Num Layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedded dim for attention')
    parser.add_argument('--feat_drop', type=float, default=0.6, help='feat_dropout')
    parser.add_argument('--attn_drop', type=float, default=0.6, help='attn_dropout')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--temp', type=float, default=1.0, help='Temp for distilling')
    parser.add_argument('--mlp_layers', type=int, default=1, help='Add feature mlp/lr')
    parser.add_argument('--grad', type=int, default=0, help='Output Feature grad')
    parser.add_argument('--asstype', type=int, default=0, help='Different assistant teacher. 0. nasty 1. reborn')

    parser.add_argument('--att', action='store_false', default=True, help='Output attention or not')
    parser.add_argument('--layer_flag', action='store_true', default=False, help='Layer output or not')

    return parser.parse_args()


def choose_path(conf):
    if args.asstype == 0:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], 'nasty_' + conf['teacher'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    else:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], 'reborn_' + conf['teacher'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                      'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate), 'cascade')
    check_readable(cascade_dir)
    return output_dir, cascade_dir


def choose_model(conf):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        if conf['model_name'] == 'GAT':
            num_heads = 8
        else:
            num_heads = 1
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,  # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=features.shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(conf['device'])
    elif conf['model_name'] == 'MoNet':
        model = MoNet(g=G,
                      in_feats=features.shape[1],
                      n_hidden=64,
                      out_feats=labels.max().item() + 1,
                      n_layers=1,
                      dim=2,
                      n_kernels=3,
                      dropout=0.7).to(conf['device'])
    elif conf['model_name'] == 'SGC':
        model = SGConv(in_feats=features.shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
            conf['layer'] = 32
            conf['hidden'] = 256
            conf['lamda'] = 0.6
            conf['dropout'] = 0.7
        elif conf['dataset'] == 'pubmed':
            conf['hidden'] = 256
            conf['lamda'] = 0.4
            conf['dropout'] = 0.5
        model = GCNII(nfeat=features.shape[1],
                      nlayers=conf['layer'],
                      nhidden=conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=conf['dropout'],
                      lamda=conf['lamda'],
                      alpha=conf['alpha'],
                      variant=False).to(conf['device'])
    return model


def distill_train(all_logits, dur, epoch):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    # print(G.ndata['alpha'])
    logp = F.log_softmax(logits, dim=1)
    # we only compute loss for labeled nodes
    # loss = F.nll_loss(logp[idx_train], labels[idx_train])
    if args.asstype == 0:
        loss = F.nll_loss(logp[idx_train], labels[idx_train]) - 0.5 * F.kl_div(logp, cas[-1])
    else:
        loss = F.kl_div(logp, cas[-1], reduction='batchmean')
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    dur.append(time.time() - t0)
    model.eval()
    if conf['model_name'] in ['GCN', 'APPNP', 'LogReg', 'MLP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        logits = model(G.ndata['feat'])[0]
    elif conf['model_name'] == 'GraphSAGE':
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'PLP':
        logits = model(G.ndata['feat'], labels_init)[0]
    logp = F.log_softmax(logits, dim=1)
    all_logits.append(logp.cpu().detach().numpy())
    # if conf['model_name'] == 'PLP':
    #     loss_val = my_loss(logp[idx_no_train], cas[-1][idx_no_train]) + F.nll_loss(logp[idx_val], labels[idx_val])
    # else:
    #     loss_val = my_loss(logp, cas[-1])
    loss_val = my_loss(logp[idx_val], cas[-1][idx_val])
    # loss_val = loss
    # loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    acc_val = accuracy(logp[idx_val], labels[idx_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    print('Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | Time(s) %.4f' % (
        epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, loss_val


def model_train(conf, model, optimizer, all_logits):
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    while epoch < conf['max_epoch']:
        acc_val, loss_val = distill_train(all_logits, dur, epoch)
        epoch += 1
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == conf['patience'] or epoch == conf['max_epoch']:
            print("Stop!!!")
            break
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))


def distill_test(conf):
    model.eval()
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, G.edata['a'] = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    preds = torch.argmax(logp, dim=1).cpu().detach()
    teacher_preds = torch.argmax(cas[-1], dim=1).cpu().detach()
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    acc_teacher_test = accuracy(cas[-1][idx_test], labels[idx_test])
    same_predict = np.count_nonzero(teacher_preds[idx_test] == preds[idx_test]) / len(idx_test)
    acc_dis = np.abs(acc_teacher_test.item() - acc_test.item())
    print("Test set results: loss= {:.4f} acc_test= {:.4f} acc_teacher_test= {:.4f} acc_dis={:.4f} same_predict= {:.4f}".format(
        loss_test.item(), acc_test.item(), acc_teacher_test.item(), acc_dis, same_predict))

    return acc_test, logp, same_predict


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    conf = get_training_config(config_path, dataset=args.dataset, teacher=args.teacher, student=args.teacher,
                               distill=True)
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    _config = get_experiment_config(config_data_path)
    conf['division_seed'] = _config['seed']
    if args.device > 0:
        conf['device'] = torch.device("cuda:"+str(args.device))
    else:
        conf['device'] = torch.device("cpu")
    if conf['model_name'] == 'PLP':
        conf = dict(conf, **args.__dict__)
    output_dir, cascade_dir = choose_path(conf)
    logger = get_logger(output_dir.joinpath('log'))
    print(conf)
    print(output_dir)
    print(cascade_dir)

    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Load data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
        load_tensor_data(conf['model_name'], conf['dataset'], args.labelrate, conf['device'])
    labels_init = initialize_label(idx_train, labels_one_hot).to(conf['device'])
    idx_no_train = torch.LongTensor(np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(conf['device'])
    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(conf['device'])
    byte_idx_train[idx_train] = True
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    G.ndata['feat'].requires_grad_()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print('Loading cascades...')
    cas = load_cascades(cascade_dir, conf['device'], final=True)
    # print('acc_teacher_test: ', accuracy(cas[-1][idx_test], labels[idx_test]))
    model = choose_model(conf)
    if conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'pubmed':
            conf['wd1'] = 0.0005
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': conf['wd1']},
            {'params': model.params2, 'weight_decay': conf['wd2']},
        ], lr=conf['learning_rate'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
    #                       weight_decay=conf['weight_decay'])

    all_logits = []
    model_train(conf, model, optimizer, all_logits)
    acc_test, logp, same_predict = distill_test(conf)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    acc_test = acc_test.cpu().item()
    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'), output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('same_predict.txt'), np.array([same_predict]), fmt='%.4f', delimiter='\t')
    if 'a' in G.edata:
        edge = torch.stack((G.edges()[0], G.edges()[1]),0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)
        att_torch = torch.FloatTensor(sp_att.todense())
        att_torch[idx_train, :] = torch.eye(len(adj))[idx_train, :]
        # k_propagate_prob = quick_matrix_pow(propagate_prob, epoch)
        k_propagate_prob = matrix_pow(att_torch, conf['num_layers'], att_torch[:, idx_train])
        sp_k_att = sp.coo_matrix(row_normalize(k_propagate_prob))
        sp.save_npz(output_dir.joinpath('k_attention_weight.npz'), sp_k_att, compressed=True)
    if 'alpha' in G.ndata:
        lr_ratio = G.ndata['alpha'].cpu().detach().numpy()
        el = G.ndata['el'].cpu().detach().numpy()
        er = G.ndata['er'].cpu().detach().numpy()
        np.savetxt(output_dir.joinpath('lr_ratio.txt'), lr_ratio, fmt='%.4f', delimiter='\t')
        np.savetxt(output_dir.joinpath('el.txt'), el, fmt='%.4f', delimiter='\t')
        np.savetxt(output_dir.joinpath('er.txt'), er, fmt='%.4f', delimiter='\t')
    if args.grad == 1:
        grad = G.ndata['feat'].grad.cpu().numpy()
        np.savetxt(output_dir.joinpath('grad.txt'), grad, fmt='%.4f', delimiter='\t')
    if conf['model_name'] == 'PLP':
        hyper = open(output_dir.joinpath('sta_log'), 'a')
        hyper.write(str(conf['num_layers'])+'\t')
        hyper.write(str(conf['mlp_layers'])+'\t')
        hyper.write(str(conf['emb_dim'])+'\t')
        hyper.write(str(conf['feat_drop'])+'\t')
        hyper.write(str(conf['attn_drop'])+'\t')
        hyper.write(str(conf['learning_rate'])+'\t')
        hyper.write(str(conf['weight_decay'])+'\t')
        hyper.write('%.4f\n' % acc_test)
        hyper.close()
