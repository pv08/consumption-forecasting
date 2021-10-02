import torch as T
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import LinBnDrop, SigmoidRange
from tsai.imports import *
from tsai.models.layers import *


class MLP(Module):
    def __init__(self, device, c_in, c_out, seq_len, layers=[500,500,500], ps=[0.1, 0.2, 0.2], act=nn.ReLU(inplace=True),
                 use_bn=False, bn_final=False, lin_first=False, fc_dropout=0., y_range=None):
        layers, ps = L(layers), L(ps)
        if len(ps) <= 1: ps = ps * len(layers)
        assert len(layers) == len(ps), '#layers and #ps must match'
        self.flatten = Reshape(-1)
        nf = [c_in * seq_len] + layers
        self.mlp = nn.ModuleList()
        for i in range(len(layers)): self.mlp.append(LinBnDrop(nf[i], nf[i+1], bn=use_bn, p=ps[i], act=act, lin_first=lin_first))
        _head = [LinBnDrop(nf[-1], c_out, bn=bn_final, p=fc_dropout)]
        if y_range is not None: _head.append(SigmoidRange(*y_range))
        self.head = nn.Sequential(*_head)
        self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        for mlp in self.mlp: x = mlp(x)
        return self.head(x)



class LinearModel(nn.Module):
    def __init__(self, device, n_features, n_hidden, activation_function):
        super(LinearModel, self).__init__()
        self.device = device
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activation_function = activation_function

        self.fc1 = nn.Linear(self.n_features, n_hidden*2)
        self.fc2 = nn.Linear(n_hidden * 2, n_hidden)
        self.regressor = nn.Linear(n_hidden, 1)
        self.to(self.device)

    def forward(self, x):
        # [batch, sqq, feature] - [1, 60, 13] -> [1, 60, 1]
        if self.activation_function == 'relu':
            layer1 = F.relu(self.fc1(x))
            layer2 = F.relu(self.fc2(layer1))
            layer3 = self.regressor(layer2)
        if self.activation_function == 'sigmoid':
            layer1 = T.sigmoid(self.fc1(x))
            layer2 = T.sigmoid(self.fc2(layer1))
            layer3 = self.regressor(layer2)
        return layer3[:,0,:]


