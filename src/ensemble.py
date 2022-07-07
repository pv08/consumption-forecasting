import pytorch_lightning as pl
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import mkdir_if_not_exists, _regressor_ensemble_class_dict, write_test_json
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler

class PecanEnsemble(PecanWrapper):
    def __init__(self, args):
        super(PecanEnsemble, self).__init__(args)

        self.ensemble_name = self.args.ensemble_method
        for model in self.args.ensemble_models:
            self.ensemble_name += '-' + model
        mkdir_if_not_exists(
            f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}')
        mkdir_if_not_exists(
            f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}')
        mkdir_if_not_exists(
            f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}/best/')
        mkdir_if_not_exists(
            f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}/epochs/')

        self.best_ckpt_location = f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}/best/'
        self.best_ckpt_filename = f'best-{self.ensemble_name}-chpkt-pecanstreet-participant-id-{self.args.participant_id}' + "_{epoch:03d}"

        self.every_ckpt_location = f'lib/ckpts/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}/epochs/'
        self.every_ckpt_filename = f'{self.ensemble_name}-chpkt-pecanstreet-participant-id-{self.args.participant_id}' + "_{epoch:03d}"

        mkdir_if_not_exists(
            f'lib/log/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}')

        local_logger_dir = f'lib/log/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/{self.ensemble_name}'

        master_logger_model = self.args.model
        local_logger_dir = f'lib/log/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/'

        self.args.teacher_ckpt = self.resume_ckpt
        self.regressor = _regressor_ensemble_class_dict(self.args)

        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.best_ckpt_location,
            filename=self.best_ckpt_filename,
            save_top_k=1,
            verbose=True,
            monitor="val/loss_epoch",
            mode='min'
        )
        every_checkpoint_callback = ModelCheckpoint(
            dirpath=self.every_ckpt_location,
            filename=self.every_ckpt_filename,
            save_top_k=-1,
            every_n_epochs=1,
            verbose=True,
            monitor="val/loss_epoch",
            mode='min'
        )

        self.callbacks.append(best_checkpoint_callback)
        self.callbacks.append(every_checkpoint_callback)


        if self.args.early_stopping:
            self.callbacks.append(EarlyStopping(monitor="val/loss_epoch",
                                           patience=self.args.patience))

        if self.resume_ckpt is not None:
            logger_name = f"TrainerPL_{master_logger_model}_{self.args.activation_fn}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}_{self.args.lr}_resume_from_{self.number_last_epoch}"

        else:
            logger_name = f"TrainerPL_{master_logger_model}_{self.args.activation_fn}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}_{self.args.lr}"

        self.logger_master = WandbLogger(project='pecanstreet',
                             tags=[self.args.model, self.args.participant_id, self.args.activation_fn, "multi-step regressor", self.args.task],
                             offline=False,
                             name=logger_name,
                             config=self.args)

        self.logger_slave = CSVLogger(local_logger_dir, name = logger_name)



        self.logger_master.log_hyperparams(self.regressor.hparams)
        self.logger_master.watch(self.regressor, log='all')
        self.logger_slave.log_hyperparams(self.regressor.hparams)


        self.trainer = pl.Trainer(
            logger=[self.logger_master, self.logger_slave],
            enable_checkpointing=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=60,
            resume_from_checkpoint=self.resume_ckpt,
            accumulate_grad_batches=1
        )




    def ensemble(self):
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=60
        )

        result = trainer.fit(self.regressor, self.data_module.test_dataloader())
        write_test_json(result, self.ensemble_name, 'test', self.args.participant_id)
        write_test_json(self.regressor.predictions, self.ensemble_name, 'predict', self.args.participant_id)



