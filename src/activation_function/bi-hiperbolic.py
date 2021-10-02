import numpy as np
import math
from torch.nn.modules.activation import Module
from torch import Tensor


def bi_hiperbolic(input, t1, t2, alpha):
    lamb1 = 4*(0.5*math.tanh(alpha))*math.sqrt((1/16) * (t1**2))
    lamb2 = 4 * (0.5 * math.tanh(alpha)) * math.sqrt((1 / 16) * (t2 ** 2))
    return np.sqrt((lamb1**2) * (input + (1/4*lamb1)))

class BiHiperbolic(Module):
    def forward(self, input: Tensor) -> Tensor:
        return bi_hiperbolic(input, 1,2,180)
