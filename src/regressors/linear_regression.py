import torch.nn as nn
import torch.optim as optim
import torch as T
from src.models.linear import LinearModel, MLP
from src.regressors.basic_regressor import BasicRegressor

class ConsumptionLinearRegressor(BasicRegressor):
    def __init__(self, device, n_features, lr, n_hidden, activation_function, scaler = None):
        super().__init__(scaler)
        self.model = LinearModel(device, n_features,
                                 n_hidden, activation_function)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)

class ConsumptionMLPRegressor(BasicRegressor):
    def __init__(self, device, n_features, sequence_length, lr, scaler = None):
        super().__init__(scaler)
        self.model = MLP(device, n_features, 1,
                                 sequence_length)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)