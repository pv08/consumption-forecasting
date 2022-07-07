import torch.nn as nn
import torch
import torch.optim as optim
from src.models.transformer import TransformerModel
from src.models.tst import TSTModel
from src.regressors.basic_regressor import BasicRegressor

class ConsumptionTransformerRegressor(BasicRegressor):
    def __init__(self, device, n_features, d_model, n_head, d_ffn, dropout, n_layers,
                 lr,  activation_function, scaler = None):
        super(ConsumptionTransformerRegressor, self).__init__(scaler)
        self.model = TransformerModel(device=device, c_in=n_features, c_out=1, d_model=d_model,
                                      n_head=n_head, d_ffn=d_ffn, dropout=dropout, n_layers=n_layers,
                                      activation=activation_function)
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()
        self.lr = lr

        self.n_features = n_features

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)


class ConsumptionTSTRegressor(BasicRegressor):
    def __init__(self, device, n_features, seq_len, max_seq_len, d_model, n_head, d_k, d_v, d_ffn, res_dropout, n_layers,
                 lr,  activation_function, fc_dropout, scaler = None) -> object:
        super(ConsumptionTSTRegressor, self).__init__(scaler)
        self.model = TSTModel(device=device, c_in=n_features, c_out=1, seq_len=seq_len, max_seq_len=max_seq_len,
                 n_layers=n_layers, d_model=d_model, n_heads=n_head, d_k=d_k, d_v=d_v,
                 d_ff=d_ffn, res_dropout=res_dropout, act=activation_function, fc_dropout=fc_dropout)

        self.criterion = nn.MSELoss()
        self.lr = lr

        self.n_features = n_features

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)
