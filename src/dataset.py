import torch as T
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class PecanDataset(Dataset):
    def __init__(self, sequences, device):
        self.sequences = sequences
        self.device = device


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence = T.Tensor(sequence.to_numpy()),
            label = T.tensor(label).float()
        )

class PecanDataModule(pl.LightningDataModule):
    def __init__(self, device, shap_background_sequence, shap_test_sequence, train_sequences, test_sequences,
                 val_sequences, background_shap_bs = 512, test_shap_bs = 128 ,batch_size = 8, num_workers = 6,
                 pin_memory = True):
        super(PecanDataModule, self).__init__()
        self.shap_background_sequence = shap_background_sequence
        self.shap_test_sequence = shap_test_sequence

        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.val_sequences = val_sequences

        self.background_shap_bs = background_shap_bs
        self.test_shap_bs = test_shap_bs

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device


    def setup(self):
        self.shap_background_sequence = PecanDataset(self.shap_background_sequence, self.device)
        self.shap_test_sequence = PecanDataset(self.shap_test_sequence, self.device)

        self.train_sequences = PecanDataset(self.train_sequences, self.device)
        self.test_sequences = PecanDataset(self.test_sequences, self.device)
        self.val_sequences = PecanDataset(self.val_sequences, self.device)

    def train_dataloader(self):
        return DataLoader(
            self.train_sequences,
            batch_size=self.batch_size,
            shuffle = False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_sequences,
            batch_size=1,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_sequences,
            batch_size=1,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory=self.pin_memory
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def _shap_background_dataloader(self):
        return DataLoader(
            self.shap_background_sequence,
            batch_size=self.background_shap_bs,
            shuffle = False,
            num_workers = 0,
            pin_memory=False
        )

    def _shap_test_dataloader(self):
        return DataLoader(
            self.shap_test_sequence,
            batch_size=self.test_shap_bs,
            shuffle = False,
            num_workers = 0,
            pin_memory=False
        )
