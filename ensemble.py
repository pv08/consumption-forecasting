import argparse
import torch.nn as nn
import torch.optim as optim
import torch as T
import pytorch_lightning as pl
from src.utils.functions import _get_multiples_best_epochs, _get_resume_and_best_epoch
from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.dataset import PecanDataModule
from src.models.gru import GRUModel
from src.models.rnn import RNNModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from sklearn.metrics import r2_score
from torchmetrics import ExplainedVariance
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import WeightedMeanAbsolutePercentageError
from torchmetrics import SymmetricMeanAbsolutePercentageError

from torchmetrics import MeanSquaredError
from torchmetrics import MeanSquaredLogError

from torchmetrics import PearsonCorrCoef
from torchmetrics import SpearmanCorrCoef

from torchmetrics import R2Score
from torchmetrics import TweedieDevianceScore

import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from src.regressors.lstm_regressor import ConsumptionLSTMRegressor
from src.regressors.gru_regression import ConsumptionGRURegressor
from src.regressors.transformer_regressor import ConsumptionTransformerRegressor


class EnsembleModel(pl.LightningModule):
    def __init__(self, args):
        super(EnsembleModel, self).__init__()
        teacher_ckpt = _get_resume_and_best_epoch('test', '661_test_30_pca', 'sigmoid', 'RNN')

        self.teacher_model = ConsumptionRNNRegressor(device=args.device, n_features=args.n_features, lr=1e-5,
                                                     n_hidden=256, n_layers=3, dropout=0.3, activation_function='sigmoid')

        gru_model = ConsumptionGRURegressor(device=args.device, n_features=args.n_features, lr=1e-5,
                                                     n_hidden=256, n_layers=3, dropout=0.3, activation_function='sigmoid')
        lstm_model = ConsumptionLSTMRegressor(device=args.device, n_features=args.n_features, lr=1e-5, n_hidden=256, n_layers=3, dropout=0.3,
                                              activation_function='sigmoid', bidirectional=False)

        self.student_models = nn.ModuleList([gru_model, lstm_model, self.teacher_model])
        self.criterion = nn.MSELoss()

        self.explained_var = ExplainedVariance()
        self.MAE = MeanAbsoluteError()
        self.MAPE = MeanAbsolutePercentageError()
        self.SMAPE = SymmetricMeanAbsolutePercentageError()
        self.WMAPE = WeightedMeanAbsolutePercentageError()
        self.MSE = MeanSquaredError()
        self.RMSE = MeanSquaredError(squared=False)
        self.MSLE = MeanSquaredLogError()
        self.pearson_coef = PearsonCorrCoef()
        self.tweedie_dev = TweedieDevianceScore()  # power 0 for normal distribution


    def forward(self, ensemble, labels=None):
        outputs = [model['mean'] for model in ensemble]
        ensemble_output = T.stack(outputs, 0).mean(0)
        loss = 0
        if labels is not None:
            loss = self.criterion(ensemble_output, labels.unsqueeze(dim=1))
        return loss, ensemble_output

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        ensemble_model_dict = []
        for model in self.student_models:
            model_loss, model_outputs = model.training_step(batch, batch_idx)
            ensemble_model_dict.append({
                'model': type(model).__class__,
                'outputs': model_outputs
            })
        rank_df = pd.DataFrame(ensemble_model_dict)
        rank_df['score_rank'] = rank_df['loss'].rank(ascending=True)
        rank_df['weights'] = rank_df['score_rank'] / sum(np.array(rank_df['score_rank']))
        rank_df['mean'] = rank_df['outputs'] * rank_df['weights']
        ensemble = rank_df.to_dict('records')

        loss, outputs = self(ensemble, labels)
        self.log("train/loss_epoch", loss, prog_bar=True, logger=True)

        return loss, outputs

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        ensemble_model_dict = []
        for model in self.student_models:
            model_loss, model_outputs = model.validation_step(batch, batch_idx)
            ensemble_model_dict.append({
                'model': type(model).__name__,
                'outputs': model_outputs,
                'loss': model_loss
            })
        rank_df = pd.DataFrame(ensemble_model_dict)
        rank_df['score_rank'] = rank_df['loss'].rank(ascending=False)
        rank_df['weights'] = rank_df['score_rank'] / sum(np.array(rank_df['score_rank']))
        rank_df['mean'] = rank_df['outputs'] * rank_df['weights']
        ensemble = rank_df.to_dict('records')

        loss, outputs = self(ensemble, labels)

        self.log("val/loss_epoch", loss, prog_bar=True, logger=True)
        self.log("val/val_mae", self.MAE(outputs[:, 0], labels), prog_bar=True, logger=True)
        self.log("val/val_mse", self.MSE(outputs[:, 0], labels), prog_bar=True, logger=True)
        self.log("val/rmse", self.RMSE(outputs[:, 0], labels), prog_bar=True, logger=True)
        self.log("val/mape", self.MAPE(outputs[:, 0], labels), prog_bar=True, logger=True)

        return loss, outputs

    def configure_optimizers(self):
        parameters = self.student_models.parameters()

        optimizer = optim.AdamW(parameters, lr=1e-5)

        scheduler = {'scheduler': T.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,
                                                                  steps_per_epoch=90360 // 16,
                                                                  epochs=5),
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    pecan_dataset = PecanParticipantPreProcessing('661_test_30_pca', 'data/participants_data/1min/', 60, 'train')
    train_sequences, test_sequences, val_sequences = pecan_dataset.get_sequences()

    args.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    args.n_features = pecan_dataset.get_n_features()
    args.scaler = pecan_dataset.scaler
    data_module = PecanDataModule(
        device=args.device,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        val_sequences=val_sequences,
        batch_size=16,
        num_workers=0,
        pin_memory=True
    )
    data_module.setup()

    regressor = EnsembleModel(args)
    callbacks = []

    best_checkpoint_callback = ModelCheckpoint(
        dirpath='/',
        filename='best_ckpt_filename_ensemble',
        save_top_k=1,
        verbose=True,
        monitor="val/loss_epoch",
        mode='min'
    )
    every_checkpoint_callback = ModelCheckpoint(
        dirpath='/',
        filename='every_ckpt_filename_ensemble',
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
        monitor="val/loss_epoch",
        mode='min'
    )

    callbacks.append(best_checkpoint_callback)
    callbacks.append(every_checkpoint_callback)

    logger_master = WandbLogger(project='pecanstreet',
                                tags=['ensemble', 'test'],
                                offline=False,
                                name='ensemble_test')

    logger_slave = CSVLogger('/', name='ensemble_test')

    logger_master.log_hyperparams(regressor.hparams)
    logger_master.watch(regressor, log='all')
    logger_slave.log_hyperparams(regressor.hparams)

    trainer = pl.Trainer(
        logger=[logger_master, logger_slave],
        enable_checkpointing=True,
        callbacks=callbacks,
        max_epochs=5,
        gpus=1,
        progress_bar_refresh_rate=60,
        accumulate_grad_batches=1
    )

    trainer.fit(regressor, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == "__main__":
    main()
