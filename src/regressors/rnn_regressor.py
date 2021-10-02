import torch.nn as nn
import torch.optim as optim
from src.models.rnn import RNNModel
from src.regressors.basic_regressor import BasicRegressor

class ConsumptionRNNRegressor(BasicRegressor):
    def __init__(self, device, n_features, lr, n_hidden, n_layers, dropout, activation_function, scaler = None):
        super(ConsumptionRNNRegressor, self).__init__(scaler)
        self.model = RNNModel(device, n_features, n_hidden, n_layers, dropout, activation_function)
        self.criterion = nn.MSELoss()
        self.lr = lr

        self.n_features = n_features
        self.n_hidden = n_hidden

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)