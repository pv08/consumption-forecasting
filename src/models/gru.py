import torch.nn as nn
import torch.nn.functional as F
import torch as T


class GRUModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, layer_dim, dropout_prob, activation_function):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function


        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.regressor = nn.Linear(hidden_dim * 2, 1)

        self.device = device
        self.to(self.device)


    def forward(self, x):
        self.gru.flatten_parameters()

        _, hidden = self.gru(x)

        out = hidden[-1]

        if self.activation_function == 'relu':
            layer1 = F.relu(self.fc1(out))
        elif self.activation_function == 'sigmoid':
            layer1 = T.sigmoid(self.fc1(out))

        return self.regressor(layer1)
