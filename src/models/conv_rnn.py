import torch.nn as nn
import torch
class ConvRNN(nn.Module):
    def __init__(self, device, input_dim, timesteps, output_dim=1, kernel_size1=7, kernel_size2=5, kernel_size3=3,
                 n_channels1=128, n_channels2=128, n_channels3=128, n_units1=128, n_units2=128, n_units3=128):
        super().__init__()
        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.avg_pool2 = nn.AvgPool1d(4, 4)
        self.conv11 = nn.Conv1d(input_dim, n_channels1, kernel_size=kernel_size1)
        self.conv12 = nn.Conv1d(n_channels1, n_channels1, kernel_size=kernel_size1)
        self.conv21 = nn.Conv1d(input_dim, n_channels2, kernel_size=kernel_size2)
        self.conv22 = nn.Conv1d(n_channels2, n_channels2, kernel_size=kernel_size2)
        self.conv31 = nn.Conv1d(input_dim, n_channels3, kernel_size=kernel_size3)
        self.conv32 = nn.Conv1d(n_channels3, n_channels3, kernel_size=kernel_size3)
        self.gru1 = nn.GRU(n_channels1, n_units1, batch_first=True)
        self.gru2 = nn.GRU(n_channels2, n_units2, batch_first=True)
        self.gru3 = nn.GRU(n_channels3, n_units3, batch_first=True)
        self.linear1 = nn.Linear(n_units1 + n_units2 + n_units3, output_dim)
        self.linear2 = nn.Linear(input_dim * timesteps, output_dim)
        self.zp11 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)
        self.zp12 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)
        self.zp21 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)
        self.zp22 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)
        self.zp31 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)
        self.zp32 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # line1
        y1 = self.zp11(x)
        y1 = torch.relu(self.conv11(y1))
        y1 = self.zp12(y1)
        y1 = torch.relu(self.conv12(y1))
        y1 = y1.permute(0, 2, 1)
        out, h1 = self.gru1(y1)
        # line2
        y2 = self.avg_pool1(x)
        y2 = self.zp21(y2)
        y2 = torch.relu(self.conv21(y2))
        y2 = self.zp22(y2)
        y2 = torch.relu(self.conv22(y2))
        y2 = y2.permute(0, 2, 1)
        out, h2 = self.gru2(y2)
        # line3
        y3 = self.avg_pool2(x)
        y3 = self.zp31(y3)
        y3 = torch.relu(self.conv31(y3))
        y3 = self.zp32(y3)
        y3 = torch.relu(self.conv32(y3))
        y3 = y3.permute(0, 2, 1)
        out, h3 = self.gru3(y3)
        h = torch.cat([h1[-1], h2[-1], h3[-1]], dim=1)
        out1 = self.linear1(h)
        out2 = self.linear2(x.contiguous().view(x.shape[0], -1))
        out = out1 + out2
        return out