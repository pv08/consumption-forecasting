import os
from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.hue_dataset.hue_preprocessing import HUEPreProcessing
from src.dataset import PecanDataModule
from src.utils.functions import mkdir_if_not_exists
from src.regressors.linear_regression import ConsumptionLinearRegressor, ConsumptionMLPRegressor
from src.regressors.lstm_regressor import ConsumptionLSTMRegressor
from src.regressors.gru_regression import ConsumptionGRURegressor
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from src.regressors.conv_rnn_regressor import ConsumptionConvRNNRegressor
from src.regressors.transformer_regressor import ConsumptionTransformerRegressor, ConsumptionTSTRegressor
from src.regressors.fcn_regressor import ConsumptionFCNRegressor
from src.regressors.tcn_regressor import ConsumptionTCNRegressor
from src.regressors.resnet_regressor import ConsumptionResNetRegressor

import pytorch_lightning as pl

class PecanWrapper:
    def __init__(self, args):
        self.args = args
        pl.seed_everything(args.seed)
        self.mkDefaultDirs()


        self.callbacks = []
        assert self.args.dataset in ['Pecanstreet', 'HUE'], "[?] - Dataset option not recognized. Select Pecanstreet or HUE"
        if self.args.dataset == 'Pecanstreet':
            self.dataset =  PecanParticipantPreProcessing(root_path=self.args.root_path, id=self.args.participant_id,
                                                          sequence_length=self.args.sequence_length, task='train',
                                                          resolution=self.args.resolution, type=self.args.data_type)
        elif self.args.dataset == 'HUE':
            self.dataset = HUEPreProcessing(root_path=self.args.root_path, id=self.args.participant_id,
                                            debug=self.args.debug, debug_percent=self.args.debug_percent,
                                            sequence_length=self.args.sequence_length)

        self.train_sequences, self.test_sequences, self.val_sequences = self.dataset.train_sequences, self.dataset.test_sequences, self.dataset.val_sequences
        self.args.n_features = self.dataset.n_features

        self.args.scaler = self.dataset.scaler

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

    def mkDefaultDirs(self):
        self.task = 'single-step' if self.args.output_length == 1 else 'multi-step'
        #Create etc folder for directory componentes
        mkdir_if_not_exists('etc/')
        # Create checkpoints folders for training
        mkdir_if_not_exists('etc/ckpts/')
        mkdir_if_not_exists('etc/ckpts/participants/')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}')
        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}/epochs')
        self.every_ckpt_location = f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}/epochs'
        self.every_ckpt_filename = f'{self.task}-{self.args.model}-ckpt-{self.args.dataset}-participant-id-{self.args.participant_id}' + "_{epoch:03d}"

        mkdir_if_not_exists(f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}/best')
        self.best_ckpt_location = f'etc/ckpts/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}/best'
        self.best_ckpt_filename = f'best-{self.task}-{self.args.model}-ckpt-{self.args.dataset}-participant-id-{self.args.participant_id}' + "_{epoch:03d}"

        #Create log folder for training
        mkdir_if_not_exists('etc/log/')
        mkdir_if_not_exists('etc/log/participants/')
        mkdir_if_not_exists(f'etc/log/participants/{self.args.dataset}/')
        mkdir_if_not_exists(f'etc/log/participants/{self.args.dataset}/{self.task}')
        mkdir_if_not_exists(f'etc/log/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/log/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}')
        mkdir_if_not_exists(f'etc/log/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}')
        self.local_logger_dir = f'etc/log/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}'

        #Create results folders
        mkdir_if_not_exists('etc/results/')
        mkdir_if_not_exists(f'etc/results/{self.args.dataset}')
        mkdir_if_not_exists(f'etc/results/{self.args.dataset}/{self.task}')
        mkdir_if_not_exists(f'etc/results/{self.args.dataset}/{self.task}/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/results/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}')
        mkdir_if_not_exists(f'etc/results/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}')
        self.local_result_dir = f'etc/results/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}'


        #Create img folders for validation
        mkdir_if_not_exists('etc/imgs/')
        mkdir_if_not_exists('etc/imgs/participants')
        mkdir_if_not_exists(f'etc/imgs/participants/{self.args.dataset}')
        mkdir_if_not_exists(f'etc/imgs/participants/{self.args.dataset}/{self.task}')
        mkdir_if_not_exists(f'etc/imgs/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/imgs/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}')
        mkdir_if_not_exists(f'etc/imgs/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}')
        self.local_imgs_dir = f'etc/imgs/participants/{self.args.dataset}/{self.task}/{self.args.participant_id}/{self.args.resolution}/{self.args.model}'
        
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def validator(self):
        raise NotImplementedError

    def ensemble(self):
        raise NotImplementedError

    def reports(self):
        raise NotImplementedError

    @staticmethod
    def _get_trained_regressor_model(args, ckpt, scaler):
        model = None
        if args.model == "LSTM":
            model = ConsumptionLSTMRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                  strict=False,
                                                                  device=args.device,
                                                                  n_features=args.n_features,
                                                                  lr=args.lr,
                                                                  n_hidden=args.n_hidden,
                                                                  n_layers=args.n_layers,
                                                                  dropout=args.dropout,
                                                                  activation_function=args.activation_fn,
                                                                  bidirectional=args.bidirectional, scaler=scaler)
        elif args.model == "RNN":
            model = ConsumptionRNNRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                 strict=False,
                                                                 scaler=scaler,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 lr=args.lr,
                                                                 n_hidden=args.n_hidden,
                                                                 n_layers=args.n_layers,
                                                                 dropout=args.dropout,
                                                                 activation_function=args.activation_fn)
        elif args.model == "GRU":
            model = ConsumptionGRURegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                 strict=False,
                                                                 scaler=scaler,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 lr=args.lr,
                                                                 n_hidden=args.n_hidden,
                                                                 n_layers=args.n_layers,
                                                                 dropout=args.dropout,
                                                                 activation_function=args.activation_fn)
        elif args.model == "Linear":
            model = ConsumptionLinearRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                    strict=False,
                                                                    scaler=scaler,
                                                                    device=args.device,
                                                                    n_features=args.n_features,
                                                                    lr=args.lr,
                                                                    n_hidden=args.n_hidden,
                                                                    activation_function=args.activation_fn)
        elif args.model == "MLP":
            model = ConsumptionMLPRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                 strict=False,
                                                                 scaler=scaler,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 sequence_length=args.sequence_length,
                                                                 lr=args.lr)
        elif args.model == "FCN":
            model = ConsumptionFCNRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                 strict=False,
                                                                 scaler=scaler,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 lr=args.lr,
                                                                 activation_function=args.activation_fn)
        elif args.model == "TCN":
            model = ConsumptionTCNRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                 strict=False,
                                                                 scaler=scaler,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 lr=args.lr,
                                                                 activation_function=args.activation_fn)
        elif args.model == "ResNet":
            model = ConsumptionResNetRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                    strict=False,
                                                                    scaler=scaler,
                                                                    device=args.device,
                                                                    n_features=args.n_features,
                                                                    lr=args.lr,
                                                                    activation_function=args.activation_fn)
        elif args.model == "ConvRNN":
            model = ConsumptionConvRNNRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                     strict=False,
                                                                     scaler=scaler,
                                                                     device=args.device,
                                                                     n_features=args.n_features,
                                                                     time_steps=args.sequence_length,
                                                                     lr=args.lr,
                                                                     activation_function=args.activation_fn)
        elif args.model == "Transformer":
            model = ConsumptionTransformerRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                                         strict=False,
                                                                         scaler=scaler,
                                                                         device=args.device,
                                                                         n_features=args.n_features,
                                                                         d_model=args.d_model,
                                                                         n_head=args.n_head,
                                                                         d_ffn=args.d_ffn,
                                                                         dropout=args.dropout,
                                                                         n_layers=args.n_layers,
                                                                         lr=args.lr,
                                                                         activation_function=args.tst_activation_fn)
        elif args.model == "TST":
            model = ConsumptionTSTRegressor.load_from_checkpoint(checkpoint_path=ckpt, scaler=scaler,
                                                                 strict=False,
                                                                 device=args.device,
                                                                 n_features=args.n_features,
                                                                 seq_len=args.sequence_length,
                                                                 max_seq_len=args.max_seq_len, d_model=args.d_model,
                                                                 n_head=args.n_head,
                                                                 d_k=args.d_k, d_v=args.d_v, d_ffn=args.d_ffn,
                                                                 res_dropout=args.res_dropout,
                                                                 n_layers=args.n_layers, lr=args.lr,
                                                                 activation_function=args.tst_activation_fn,
                                                                 fc_dropout=args.fc_dropout)

        else:
            raise NotImplementedError(f"[?] - Model not implemented yet")
        return model



    @staticmethod
    def _get_regressor_model_(args):
        model = None
        if args.model == "LSTM":
            model = ConsumptionLSTMRegressor(device=args.device,
                                             n_features=args.n_features,
                                             lr=args.lr,
                                             n_hidden=args.n_hidden,
                                             n_layers=args.n_layers,
                                             dropout=args.dropout,
                                             activation_function=args.activation_fn,
                                             bidirectional=args.bidirectional)
        elif args.model == "RNN":
            model = ConsumptionRNNRegressor(device=args.device,
                                            n_features=args.n_features,
                                            lr=args.lr,
                                            n_hidden=args.n_hidden,
                                            n_layers=args.n_layers,
                                            dropout=args.dropout,
                                            activation_function=args.activation_fn)
        elif args.model == "GRU":
            model = ConsumptionGRURegressor(device=args.device,
                                            n_features=args.n_features,
                                            lr=args.lr,
                                            n_hidden=args.n_hidden,
                                            n_layers=args.n_layers,
                                            dropout=args.dropout,
                                            activation_function=args.activation_fn)
        elif args.model == "Linear":
            model = ConsumptionLinearRegressor(device=args.device,
                                               n_features=args.n_features,
                                               lr=args.lr,
                                               n_hidden=args.n_hidden,
                                               activation_function=args.activation_fn)
        elif args.model == "MLP":
            model = ConsumptionMLPRegressor(device=args.device,
                                            n_features=args.n_features,
                                            sequence_length=args.sequence_length,
                                            lr=args.lr)
        elif args.model == "FCN":
            model = ConsumptionFCNRegressor(device=args.device,
                                            n_features=args.n_features,
                                            lr=args.lr, activation_function=args.activation_fn)
        elif args.model == "TCN":
            model = ConsumptionTCNRegressor(device=args.device,
                                            n_features=args.n_features,
                                            lr=args.lr, activation_function=args.activation_fn)
        elif args.model == "ResNet":
            model = ConsumptionResNetRegressor(device=args.device,
                                               n_features=args.n_features,
                                               lr=args.lr, activation_function=args.activation_fn)
        elif args.model == "ConvRNN":
            model = ConsumptionConvRNNRegressor(device=args.device,
                                                n_features=args.n_features,
                                                time_steps=args.sequence_length,
                                                lr=args.lr, activation_function=args.activation_fn)
        elif args.model == "Transformer":
            model = ConsumptionTransformerRegressor(device=args.device,
                                                    n_features=args.n_features,
                                                    d_model=args.d_model,
                                                    n_head=args.n_head,
                                                    d_ffn=args.d_ffn,
                                                    dropout=args.dropout,
                                                    n_layers=args.n_layers,
                                                    lr=args.lr,
                                                    activation_function=args.tst_activation_fn)
        elif args.model == "TST":
            model = ConsumptionTSTRegressor(device=args.device, n_features=args.n_features,
                                            seq_len=args.sequence_length,
                                            max_seq_len=args.max_seq_len, d_model=args.d_model, n_head=args.n_head,
                                            d_k=args.d_k, d_v=args.d_v, d_ffn=args.d_ffn, res_dropout=args.res_dropout,
                                            n_layers=args.n_layers, lr=args.lr,
                                            activation_function=args.tst_activation_fn,
                                            fc_dropout=args.fc_dropout)
        else:
            raise NotImplementedError(f"[?] - Model not implemented yet")

        return model


    @staticmethod
    def get_epoch_trained(path):
        resume_ckpt = None
        number_last_epoch = None
        try:
            list_epochs = next(os.walk(path))[2]
        finally:
            if len(list_epochs) > 0:
                last_epoch = list_epochs[len(list_epochs) - 1]
                number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
                print(f"[!] - Last Epoch loaded - {number_last_epoch}")
                resume_ckpt = f'{path}/{last_epoch}'
        return resume_ckpt, number_last_epoch

    @staticmethod
    def get_best_epoch_trained(participant, activation_fn, model):
        resume_ckpt = None
        number_last_epoch = None
        try:
            list_best_epochs = next(os.walk(
                f'etc/ckpts/participants/{participant}/{activation_fn}/{model}/best/'))[2]
        finally:
            if len(list_best_epochs) > 0:
                last_epoch = list_best_epochs[len(list_best_epochs) - 1]
                number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
                print(f"[!] - Best Epoch loaded - {number_last_epoch}")
                resume_ckpt = f'etc/ckpts/participants/{participant}/{activation_fn}/{model}/best/{last_epoch}'
        return resume_ckpt, number_last_epoch

    def _get_multiples_best_epochs(self):
        ckpt_list = []
        try:
            for ensemble_models in self.args.ensemble_models:
                ckpt, _ = self.get_best_epoch_trained(self.args.participant_id, self.args.activation_fn, ensemble_models)
                ckpt_list.append(ckpt)
        finally:
            return ckpt_list






















