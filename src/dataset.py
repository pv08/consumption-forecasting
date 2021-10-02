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
    def __init__(self, device, train_sequences, test_sequences, val_sequences, batch_size = 8, num_workers = 6):
        super(PecanDataModule, self).__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device


    def setup(self):
        self.train_sequences = PecanDataset(self.train_sequences, self.device)
        self.test_sequences = PecanDataset(self.test_sequences, self.device)
        self.val_sequences = PecanDataset(self.val_sequences, self.device)

    def train_dataloader(self):
        return DataLoader(
            self.train_sequences,
            batch_size=self.batch_size,
            shuffle = False,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_sequences,
            batch_size=1,
            shuffle = False,
            num_workers = self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_sequences,
            batch_size=1,
            shuffle = False,
            num_workers = self.num_workers
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)
