import numpy as np
import torch


def accuracy(output, labels, details=False, hop_idx=None):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()
    if details:
        hop_num = np.bincount(hop_idx, minlength=7)
        true_idx = np.array((correct > 0).nonzero().squeeze(dim=1).cpu())
        true_hop = np.bincount(hop_idx[true_idx], minlength=7)/hop_num
        return result / len(labels), true_hop
    return result / len(labels)


def eucli_dist(output, target):
    return torch.sqrt(torch.sum(torch.pow((output - target), 2)))


def my_loss(output, target, mode=0):
    # https://blog.csdn.net/nature553863/article/details/80568658
    if mode == 0:
        return eucli_dist(torch.exp(output), target)
    elif mode == 1:
        # Distilling the Knowledge in a Neural Network
        return torch.nn.BCELoss()(torch.exp(output), target)
    elif mode == 2:
        # Exploring Knowledge Distillation of Deep Neural Networks for Efficient Hardware Solutions
        return torch.nn.KLDivLoss()(output, target)
    # output = F.log_softmax(output, dim=1)
    # return torch.mean(-torch.sum(torch.mul(target, output), dim=1))
    # Cross Entropy Error Function without mean
    # return -torch.sum(torch.mul(target, output))
    # return torch.sum(F.pairwise_distance(torch.exp(output), target, p=2))
