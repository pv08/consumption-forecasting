import pytorch_lightning as pl
import numpy as np
import torch as T
from torchmetrics import ExplainedVariance
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler

class BasicRegressor(pl.LightningModule):
    def __init__(self, scaler = None):
        super().__init__()
        self.scaler = scaler

        self.val_predictions = []
        self.test_predictions = []

        self.explained_var = ExplainedVariance()
        self.MAE = MeanAbsoluteError()
        self.MAPE = MeanAbsolutePercentageError()


    def log_descaled_values(self):
        return self.outputs, self.labels

    def forward(self, x, labels = None):
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train|MSE", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val|MSE", loss, prog_bar=True, logger=True)
        self.log("val|MAE", self.MAE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("val|MAPE", self.MAPE(outputs[:,0], labels), prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)

        self.test_predictions.append(dict(
            label=labels.item(),
            model_output=outputs.item(),
            loss=loss.item()
        ))

        self.log('test|MAE', self.MAE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log('test|MAPE', self.MAPE(outputs[:,0], labels), prog_bar=True, logger=True)
        self.log("test|MSE", loss, prog_bar=True, logger=True)

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