import torch.nn as nn
import torch as T
import torch.optim as optim
import shap
from src.models.gru import GRUModel
from src.regressors.basic_regressor import BasicRegressor


class ConsumptionGRURegressor(BasicRegressor):
    def __init__(self, device, n_features, lr, n_hidden, n_layers, dropout, activation_function, scaler = None):
        super().__init__(scaler)

        self.model = GRUModel(device, n_features,
                              n_hidden, n_layers, dropout, activation_function)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

