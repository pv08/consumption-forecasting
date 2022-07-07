import torch.nn as nn
import torch.nn.functional as F
import torch as T


class RecorrentEnsembleModel(nn.Module):
    def __init__(self, device, ensemble_models: nn.ModuleList):
        super(RecorrentEnsembleModel, self).__init__()
        self.ensemble = ensemble_models
        self.regressor = nn.Softmax()

        self.to(device)


    def forward(self, x, label):
        ensemble_output = []
        for model in self.ensemble:
            _, model_output = model(x)
            ensemble_output.append(model_output)
        output = T.stack(ensemble_output).mean(0)
        return self.regressor(output)
