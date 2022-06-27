import torch.nn as nn
import torch.nn.functional as F
import torch as T


class RecorrentEnsemble(nn.Module):
    def __init__(self, device, rnn_model, lstm_model, gru_model, update_weights):
        super(RecorrentEnsemble, self).__init__()
        self.rnn_model = rnn_model
        self.lstm_model = lstm_model
        self.gru_model = gru_model

        self.rnn_w_model = 1
        self.lstm_w_model = 1
        self.gru_w_model = 1

        self.voting = update_weights

        self.regressor = nn.Linear(1, 1)

        self.to(device)

    def update_weights(self, rnn_out, lstm_out, gru_out, label):
        self.rnn_w_model = label/rnn_out
        self.lstm_w_model = label/lstm_out
        self.gru_w_model = label/gru_out

    def forward(self, x, label):
        _, rnn_out = self.rnn_model(x)
        _, lstm_out = self.lstm_model(x)
        _, gru_out = self.gru_model(x)

        ensemble_out = ((rnn_out * self.rnn_w_model) + (lstm_out * self.lstm_w_model) + (gru_out * self.gru_w_model))/(self.rnn_w_model + self.lstm_w_model + self.gru_w_model)

        self.update_weights(rnn_out, lstm_out, gru_out, label) if self.voting else None
        return T.sigmoid(self.regressor(ensemble_out))
