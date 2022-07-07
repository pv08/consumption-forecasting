import torch as T
import torch.nn as nn
import torch.optim as optim
import torchensemble
from src.regressors.basic_regressor import BasicRegressor
from src.models.recorrent_ensemble import RecorrentEnsembleModel
from src.optimizers.simulated_annealing import SimulatedAnnealinng, GaussianSampler


class RecorrentEnsembleRegressor(BasicRegressor):
    def __init__(self, device, ModelArray, lr,  scale = None):
        super(RecorrentEnsembleRegressor, self).__init__(scale)
        self.save_hyperparameters(ignore=['ModelArray'])
        ensemble = nn.ModuleList([regressor for regressor in ModelArray])
        self.model = RecorrentEnsembleModel(device=device, ensemble_models=ensemble)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss

    def configure_optimizers(self):
        sampler = GaussianSampler(mu=0, sigma=1)
        return SimulatedAnnealinng(self.model.parameters(), sampler=sampler)

    def get_ensemble_estimator(self, estimator, n_estimator=10, use_cuda=True, ensemble_method='GradientBoosting'):
        ensemble = None

        if ensemble_method == 'Fusion':
            ensemble = torchensemble.FusionRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'Voting':
            ensemble = torchensemble.VotingRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'Bagging':
            ensemble = torchensemble.BaggingRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'GradientBoosting':
            ensemble = torchensemble.GradientBoostingRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'NTE':
            ensemble = torchensemble.NeuralForestRegressor(n_estimators=n_estimator, cuda=use_cuda)
        # há outros parâmetros para configuração do tamanho da árvore
        elif ensemble_method == 'SE':
            ensemble = torchensemble.SnapshotEnsembleRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'AT':
            ensemble = torchensemble.AdversarialTrainingRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'FGE':
            ensemble = torchensemble.FastGeometricRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
        elif ensemble_method == 'SGB':
            ensemble = torchensemble.SoftGradientBoostingRegressor(estimator=estimator, n_estimators=n_estimator, cuda=use_cuda)
            # há outros parâmetros para configuração do tamanho da árvore

        return ensemble

