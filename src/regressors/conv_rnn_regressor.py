import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from src.models.conv_rnn import ConvRNN
from src.regressors.basic_regressor import BasicRegressor


class ConsumptionConvRNNRegressor(BasicRegressor):
    def __init__(self, device, n_features, time_steps, lr, activation_function, scaler = None):
        super().__init__(scaler)
        self.save_hyperparameters()
        self.model = ConvRNN(device, n_features,
                              time_steps)
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

