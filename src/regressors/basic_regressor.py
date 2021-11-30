import pytorch_lightning as pl
import numpy as np
import shap
import torch as T
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
from torchmetrics.regression.r2score import R2Score
from torchmetrics.regression.mean_absolute_percentage_error import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

class BasicRegressor(pl.LightningModule):
    def __init__(self, scaler = None):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = scaler
        self.val_mae = MeanAbsoluteError()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_mse = MeanSquaredError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_r2 = R2Score()

        self.outputs = []
        self.labels = []

    def log_descaled_values(self):
        return self.outputs, self.labels


    def forward(self, x, labels = None):
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train/loss_epoch", loss, prog_bar=True, logger=True)
        # self.log("train/r2_score", self.val_r2(outputs[:, 0], labels).item(), prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):

        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("val/loss_epoch", loss, prog_bar=True, logger=True)
        self.log("val/val_mae", self.val_mae(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/val_mse", self.val_mse(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/rmse", self.val_rmse(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val/mape", self.val_mape(outputs[:,0], labels), prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)


        self.log("test/test_loss", loss, prog_bar=True, logger=True)
        self.log("test/val_mae", self.val_mae(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("test/val_mse", self.val_mse(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("test/rmse", self.val_rmse(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("test/mape", self.val_mape(outputs[:,0], labels), prog_bar=True, logger=True)
        self.test_sequences = sequences

        return loss

    def on_test_end(self):

        self.model.train()
        self.test_sequences.requires_grad_()

        ig = IntegratedGradients(self.model)
        ig_attr, ig_delta = ig.attribute(self._interpretability_test, return_convergence_delta=True, n_steps=100)
        self.ig_attr = ig_attr.detach().cpu().numpy()[-1, :, :]

        gs = GradientShap(self.model)
        gs_attr_test = gs.attribute(self.test_sequences, self.baseline)
        self.gs_attr_test = gs_attr_test.detach().cpu().numpy()[-1, :, :]

        dl = DeepLift(self.model)
        dl_attr_test = dl.attribute(self.test_sequences)
        self.dl_attr_test = dl_attr_test.detach().cpu().numpy()[-1, :, :]

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