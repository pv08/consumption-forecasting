import torch.nn as nn
import torch.nn.functional as F
import torch as T
class LSTMModel(nn.Module):
    def __init__(self, device, n_features, n_hidden = 128, n_layers = 2, dropout = 0.2, activation_function = 'relu',
                 bidirectional = False):
        super(LSTMModel, self).__init__()
        self.n_hidden = n_hidden
        self.activation_function = activation_function

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        if bidirectional:
            self.fc1 = nn.Linear(n_hidden * 2, n_hidden * 4)
            self.regressor = nn.Linear(n_hidden * 4, 1)
        else:
            self.fc1 = nn.Linear(n_hidden, n_hidden * 2)
            self.regressor = nn.Linear(n_hidden * 2, 1)

        self.device = device

        self.to(self.device)


    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)

        out = hidden[-1] # [label, batch, feature] -> 3, 1, 256 -> [batch, feature]
        if self.activation_function == 'relu':
            layer1 = F.relu(self.fc1(out))
        elif self.activation_function == 'sigmoid':
            layer1 = T.sigmoid(self.fc1(out))

        return self.regressor(layer1) #- #[1, 1]

