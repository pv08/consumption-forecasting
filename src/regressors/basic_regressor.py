import pytorch_lightning as pl
import numpy as np
import shap
import torch as T
from torchmetrics import MetricCollection
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

from sklearn.preprocessing import MinMaxScaler
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

class BasicRegressor(pl.LightningModule):
    def __init__(self, scaler = None):
        super().__init__()
        self.scaler = scaler

        self.predictions = []

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


    def log_descaled_values(self):
        return self.outputs, self.labels

    def forward(self, x, labels = None):
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train/loss_epoch", loss, prog_bar=True, logger=True)

        return loss, outputs

    def validation_step(self, batch, batch_idx):

        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)


        self.log("val/loss_epoch", loss, prog_bar=True, logger=True)
        self.log("val/val_mae", self.MAE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/val_mse", self.MSE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/rmse", self.RMSE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/mape", self.MAPE(outputs[:,0], labels), prog_bar=True, logger=True)

        return loss, outputs



    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)

        self.predictions.append(dict(
            label=labels.item(),
            model_output=outputs.item(),
            loss=loss.item()
        ))

        self.log('test/MAE', self.MAE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/MAPE', self.MAPE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/SMAPE', self.SMAPE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/WMAPE', self.WMAPE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/MSE', self.MSE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/RMSE', self.RMSE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test/MSLE', self.MSLE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("test/test_loss", loss, prog_bar=True, logger=True)

        return loss


    def get_feature_importance_index(self):
        return {"IntegratedGradients": self.ig_attr, 'GradientShap': self.gs_attr_test, 'DeepLift': self.dl_attr_test}

    def _set_model_interpretability_baseline(self, train, device):
        baseline_batch = next(iter(train))
        self.baseline = baseline_batch['sequence'].to(device)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        def descale(descaler, values):
            values_2d = np.array(values)[:, np.newaxis]
            return descaler.inverse_transform(values_2d).flatten()
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)

        descaler = MinMaxScaler()
        descaler.min_, descaler.scale_ = self.scaler.min_[-1], self.scaler.scale_[-1]

        descaled_prediction = descale(descaler, [outputs[:,0].item()])
        descaled_label = descale(descaler, [labels.item()])
        self.outputs.append(descaled_prediction[0])
        self.labels.append(descaled_label[0])
        # self.log('predict/output', self.outputs, prog_bar=True, logger=True)
        # self.log('predict/label', self.labels, prog_bar=True, logger=True)
        return descaled_prediction[0], descaled_label[0]


    def configure_optimizers(self):
        raise NotImplemented

    def create_feature_importance(self):
        raise NotImplemented

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=False)