import torch.nn as nn
import torch.optim as optim
from src.regressors.basic_regressor import BasicRegressor
import torchensemble


class RecorrentEnsembleRegressor(BasicRegressor):
    def __init__(self, ModelArray, lr, ensemble_method = 'Voting', scale = None):
        super(RecorrentEnsembleRegressor, self).__init__(scale)
        self.save_hyperparameters(ignore=['ModelArray'])
        ignore_classes = [type(regressor.model).__name__ for regressor in ModelArray]
        model_list = [regressor.model for regressor in ModelArray]


        # self.save_hyperparameters()
        self.ensemble = nn.ModuleList(model_list)
        self.criterion = nn.MSELoss()

        self.model = self.get_ensemble_estimator(model_list[0], ensemble_method = ensemble_method)
        self.model.estimators_ = self.ensemble
        self.model.set_criterion(self.criterion)


        self.lr = lr

    def forward(self, x, labels = None):
        output = self.model.forward(x)
        loss = 0
        if labels is not None:
             loss = self.criterion(output, labels.unsqueeze(dim = 1))
        return loss, output

    def configure_optimizers(self):
        self.model.set_optimizer('AdamW', lr=self.lr, weight_decay=5e-4)

        return optim.AdamW(self.parameters(), lr = self.lr)

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

