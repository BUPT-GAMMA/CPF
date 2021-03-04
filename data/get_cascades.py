import pdb
import torch
import numpy as np
import os
# from pathlib import Path


def load_cascades(cascade_dir, device, trans=False, final=False):
    cas = []
    if final:
        cas.append(np.genfromtxt(cascade_dir.parent.joinpath('output.txt')))
    else:
        # cas_list = [np.transpose(np.genfromtxt(path)) for path in cascade_dir.rglob('*.txt')]
        cas_list = os.listdir(cascade_dir)
        cas_list.sort(key=lambda x: int(x[:-4]))
        cas.append(np.genfromtxt(cascade_dir.joinpath(cas_list[-1])))
        # for i in range(len(cas_list)):
        #     cas.append(np.genfromtxt(cascade_dir.joinpath(cas_list[i])))
    cas = torch.FloatTensor(cas)
    # pdb.set_trace()
    if trans:
        cas = torch.transpose(cas, 1, 2)
    cas = cas.to(device)
    return cas


def remove_overfitting_cascades(cascade_dir, patience):
    cas_list = os.listdir(cascade_dir)
    cas_list.sort(key=lambda x: int(x[:-4]))
    for i in range(patience):
        os.remove(cascade_dir.joinpath(cas_list[-1-i]))
