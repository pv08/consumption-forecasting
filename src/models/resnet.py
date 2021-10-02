from tsai.imports import *
from tsai.models.layers import *

class ResBlock(Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResNetModel(Module):
    def __init__(self, device, c_in, c_out):
        nf = 64
        kss = [7, 5, 3]
        self.permute = Permute(0, 2, 1)
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.fc = nn.Linear(nf * 2, c_out)
        self.to(device)

    def forward(self, x):
        x = self.permute(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.squeeze(self.gap(x))
        return self.fc(x)