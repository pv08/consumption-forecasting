import torch as T
from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.dataset import PecanDataset, PecanDataModule
from src.utils.functions import _get_resume_and_best_epoch, mkdir_if_not_exists, _get_multiples_best_epochs

import pytorch_lightning as pl

class PecanWrapper:
    def __init__(self, args):
        pl.seed_everything(0)

        self.args = args
        self.callbacks = []

        self.pecan_dataset = PecanParticipantPreProcessing(self.args.participant_id, self.args.root_path,
                                                           self.args.sequence_length, task=self.args.task)

        self.train_sequences, self.test_sequences, self.val_sequences = self.pecan_dataset.get_sequences()
        self.args.n_features = self.pecan_dataset.get_n_features()
        self.args.features_names = self.pecan_dataset.get_features_names()
        self.args.scaler = self.pecan_dataset.scaler

        self.data_module = PecanDataModule(
            device=self.args.device,
            train_sequences=self.train_sequences,
            test_sequences=self.test_sequences,
            val_sequences=self.val_sequences,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
        self.data_module.setup()

        mkdir_if_not_exists('lib/')
        mkdir_if_not_exists('lib/ckpts/')
        mkdir_if_not_exists('lib/ckpts/participants/')
        mkdir_if_not_exists(f'lib/ckpts/participants/{self.args.participant_id}/')
        mkdir_if_not_exists(f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/')
        mkdir_if_not_exists(f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/')

        mkdir_if_not_exists('lib/log/')
        mkdir_if_not_exists('lib/log/participants/')
        mkdir_if_not_exists(f'lib/log/participants/{self.args.participant_id}/')
        mkdir_if_not_exists(f'lib/log/participants/{self.args.participant_id}/{self.args.activation_fn}/')
        mkdir_if_not_exists(f'lib/log/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/')

        self.resume_ckpt, self.number_last_epoch = _get_resume_and_best_epoch(self.args.task, self.args.participant_id,
                                                                             self.args.activation_fn, self.args.model)


















