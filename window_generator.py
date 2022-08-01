import pandas as pd
import pytorch_lightning as pl

import torch.optim as optim
import torch as T
import torch.nn as nn

from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
from src.utils.functions import create_multi_sequences
from src.dataset import PecanDataModule

class LSTMModel(nn.Module):
    def __init__(self, device, n_features, n_out_features, n_hidden=128, n_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        self.fc1 = nn.Linear(in_features=n_hidden, out_features=n_hidden * 2)
        self.regressor = nn.Linear(in_features=n_hidden * 2, out_features=n_out_features)
        self.to(device)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        layer1 = T.sigmoid(self.fc1(out))

        return self.regressor(layer1)

class LSTMRegressor(LightningModule):
    def __init__(self, device, n_features, n_out_features, lr=1e-5):
        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters()

        self.criterion = nn.MSELoss()
        self.lr = lr

        self.model = LSTMModel(device=device, n_features=n_features, n_out_features=n_out_features)

    def forward(self, x, label=None):
        output = self.model(x)
        loss = 0

        if label is not None:
            loss = self.criterion(output, label.squeeze(dim=1))
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("train/loss_epoch", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val/loss_epoch", loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

def main():
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    features_df = pd.read_csv(f"data/participants_data/1min/features/661_test_30_all_features.csv")
    n = len(features_df)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    train_df = features_df[0: int(n * .7)]
    val_df = features_df[int(n * .7): int(n * (1.1 - .2))]
    test_df = features_df[int(n * (1.0 - .1)):]

    scaler = scaler.fit(features_df)


    train_df = pd.DataFrame(
        scaler.transform(train_df),
        index=train_df.index,
        columns=train_df.columns
    )

    test_df = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns
    )
    val_df = pd.DataFrame(
        scaler.transform(val_df),
        index=val_df.index,
        columns=val_df.columns
    )

    inputs_columns = train_df.columns.to_list()
    target_columns = inputs_columns

    multi_train_sequence = create_multi_sequences(train_df[inputs_columns].values,
                                            train_df[target_columns].values,
                                            n_steps_in=60,
                                            n_steps_out=1,
                                            n_sliding_steps=1,
                                            window_type='sliding')


    multi_val_sequence = create_multi_sequences(val_df[inputs_columns].values,
                                            val_df[target_columns].values,
                                            n_steps_in=60,
                                            n_steps_out=1,
                                            n_sliding_steps=1,
                                            window_type='sliding')

    multi_test_sequence = create_multi_sequences(test_df[inputs_columns].values,
                                            test_df[target_columns].values,
                                            n_steps_in=60,
                                            n_steps_out=1,
                                            n_sliding_steps=1,
                                            window_type='sliding')

    data_module = PecanDataModule(device = device, train_sequences=multi_train_sequence, test_sequences=multi_test_sequence,
                 val_sequences=multi_val_sequence, batch_size=16, num_workers=1)

    data_module.setup()
    best_ckpt_filename = f'best-lstm-chpkt-pecanstreet-participant-id-multi' + "_{epoch:03d}"
    every_ckpt_filename = f'every-lstm-chpkt-pecanstreet-participant-id-multi' + "_{epoch:03d}"

    best_checkpoint_callback = ModelCheckpoint(
        dirpath='test/ckpt/best',
        filename=best_ckpt_filename,
        save_top_k=1,
        verbose=True,
        monitor="val/loss_epoch",
        mode='min'
    )

    n_checkpoint_callback = ModelCheckpoint(
        dirpath='test/ckpt/every',
        filename=every_ckpt_filename,
        save_top_k=1,
        verbose=True,
        monitor="val/loss_epoch",
        mode='min'
    )


    regressor = LSTMRegressor(device=device, n_features=len(inputs_columns), n_out_features=len(target_columns))

    logger_master = WandbLogger(project='pecanstreet_multiouput',
                                     tags='test, multi_task',
                                     offline=False,
                                     name='Test_Multioutput')

    logger_slave = CSVLogger('test/log/lstm', name='lstm_log')

    trainer = pl.Trainer(max_epochs=200, gpus=1, logger=[logger_master, logger_slave],
                         enable_checkpointing=True, callbacks=[best_checkpoint_callback, n_checkpoint_callback], resume_from_checkpoint='test/ckpt/every/every-lstm-chpkt-pecanstreet-participant-id-multi_epoch=029.ckpt')

    trainer.fit(regressor, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == "__main__":
    main()
