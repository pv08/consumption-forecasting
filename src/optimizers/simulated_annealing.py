import torch as T
import torch.nn as nn
import numpy as np
import math
from torch.optim import Optimizer


class UniformSampler(object):
    def __init__(self, minval, maxval, dtype='float', cuda=True):
        self.minval = minval
        self.maxval = maxval
        self.cuda = cuda
        self.dtype_str = dtype
        dtypes = {
            'float': T.cuda.FloatTensor if cuda else T.FloatTensor,
            'int': T.cuda.IntTensor if cuda else T.IntTensor,
            'long': T.cuda.LongTensor if cuda else T.LongTensor
        }
        self.dtype = dtypes[dtype]

    def sample(self, size):
        if self.dtype_str == 'float':
            return self.dtype(*size).uniform_(self.minval, self.maxval)
        elif self.dtype_str == 'int' or self.dtype_str == 'long':
            return self.dtype(*size).random_(self.minval, self.maxval + 1)
        else:
            raise Exception('unknown dtype')


class GaussianSampler(object):
    def __init__(self, mu, sigma, dtype='float', cuda=True):
        self.mu = mu
        self.sigma = sigma
        self.cuda = cuda
        self.dtype_str = dtype
        dtypes = {
            'float': T.cuda.FloatTensor if cuda else T.FloatTensor,
            'int': T.cuda.IntTensor if cuda else T.IntTensor,
            'long': T.cuda.LongTensor if cuda else T.LongTensor
        }
        self.dtype = dtypes[dtype]

    def sample(self, size):
        rand_float = T.cuda.FloatTensor if self.cuda else T.FloatTensor
        rand_block = rand_float(*size).normal_(self.mu, self.sigma)

        if self.dtype_str == 'int' or self.dtype_str == 'long':
            rand_block = rand_block.type(self.dtype)

        return rand_block

class SimulatedAnnealinng(Optimizer):
    def __init__(self, params, sampler, tau0=1.0,
                 anneal_rate=0.0003, min_temp=1e-5,
                 anneal_every=100000, hard=False, hard_rate=0.9):
        defaults = dict(
            sampler=sampler,
            tau0=tau0,
            tau=tau0,
            anneal_rate=anneal_rate,
            min_temp=min_temp,
            anneal_every=anneal_every,
            hard=hard, hard_rate=hard_rate,
            iteration=0
        )
        super(SimulatedAnnealinng, self).__init__(params, defaults)

    @T.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        for group in self.param_groups:
            sampler = group['sampler']

            cloned_params = [p.clone() for p in group['params']]

            for p in group['params']:
                if group['iteration'] > 0 and group['iteration'] % group['anneal_every'] == 0:
                    if not group['hard']:
                        rate = -group['anneal_rate'] * group['iteration']
                        group['tau'] = np.maximum(group['tau0'] * np.exp(rate), group['min_temp'])
                    else:
                        group['tau'] = np.maximum(group['hard_rate'] * group['tau'], group['min_temp'])

                random_pertubation = group['sampler'].sample(p.data.size())
                p.data = p.data / T.norm(p.data)
                p.data.add_(random_pertubation)
                group['iteration'] += 1
            loss_perturbed = closure()
            final_loss, is_swapped = self.anneal(loss, loss_perturbed, group['tau'])
            if is_swapped:
                for p, pbkp in zip(group['params'], cloned_params):
                    p.data = pbkp.data
            return final_loss

    def anneal(self, loss, loss_perturbed, tau):
        def acceptance_prob(old, new, temp):
            return T.exp((old - new)/temp)

        if loss_perturbed.data[0] < loss.data[0]:
            return loss_perturbed, True
        else:
            ap = acceptance_prob(loss, loss_perturbed, tau)
            print(f"[!] - old = {loss.data[0]}, pert = {loss_perturbed.data[0]}, ap = {ap.data[0]}, tau = {tau}")
            if ap.data[0] > np.random.rand():
                return loss_perturbed, True
            return loss, False
