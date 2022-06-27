import torch.nn as nn
import torch.optim as optim
from src.models.lstm import LSTMModel
from src.regressors.basic_regressor import BasicRegressor

class ConsumptionLSTMRegressor(BasicRegressor):
    def __init__(self, device, n_features, lr, n_hidden, n_layers, dropout, activation_function, bidirectional, scaler = None, **kwargs):
        super().__init__(scaler)
        self.save_hyperparameters()
        self.n_features = n_features
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout


        self.model = LSTMModel(device, self.n_features, self.n_hidden, self.n_layers, self.dropout, activation_function,
                               bidirectional)
        self.criterion = nn.MSELoss()

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)