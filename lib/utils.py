import os
import numpy as np
import random
import torch
import scipy.sparse as sp

from torch_sparse import SparseTensor
from torch.utils import data
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct, correct / len(labels)


def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    # try:
    #     auc_out = roc_auc_score(labels, pos_probs)
    # except:
    #     auc_out = 0
    auc_out = roc_auc_score(labels, pos_probs)
    return auc_out


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    os.environ['PYTHONSEED'] = str(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=0)


def get_adj(feature, k=10):
    n_nodes = feature.shape[0]
    d = torch.cdist(feature, feature, p=2)
    sigma = torch.mean(d)
    sim = torch.exp(- d ** 2 / (2 * sigma ** 2)).sort()
    idx = sim.indices[:, -k:]
    wei = sim.values[:, -k:]

    assert n_nodes, k == idx.shape
    assert n_nodes, k == wei.shape

    I = torch.unsqueeze(torch.arange(n_nodes), dim=1).repeat(1, k).view(n_nodes * k).long()
    J = idx.reshape(n_nodes * k).long()
    V = wei.reshape(n_nodes * k)

    edge_weights = V.type(torch.FloatTensor)
    edge_index = SparseTensor(row=I, col=J)

    return edge_index, edge_weights
