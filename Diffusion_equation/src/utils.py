import numpy as np
import random
import torch
import torch.nn as nn
from collections import OrderedDict


class FCNN(torch.nn.Module):
    def __init__(self, layers, use_batch_norm=False, use_instance_norm=False):
        super(FCNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_list.append(
            ('activation_%d' % (self.depth - 1), torch.nn.Softplus())
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class FCNN_encoder(torch.nn.Module):
    def __init__(self, layers, use_batch_norm=False, use_instance_norm=False):
        super(FCNN_encoder, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.ReLU
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_random(X_all, N):
    """Given an array of (x,t) points, sample N points from this."""
    set_seed(0)

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled


def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh' or 'Tanh':
        return nn.Tanh()
    elif activation == 'relu' or 'ReLu':
        return nn.ReLU()
    elif activation == 'gelu' or 'GELU':
        return nn.GELU()
    else:
        print("WARNING: unknown activation function! Let's define by yourself!")
        return -1
