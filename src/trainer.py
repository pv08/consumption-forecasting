import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import mkdir_if_not_exists
from pytorch_lightning.loggers import WandbLogger, CSVLogger

class PecanTrainer(PecanWrapper):
    def __init__(self, args):
        super(PecanTrainer, self).__init__(args)

        self.resume_ckpt, self.number_last_epoch = self.get_epoch_trained(self.every_ckpt_location)

        self.master_logger_model = self.args.model
        self.regressor = self._get_regressor_model_(self.args)



    def train(self):
        #TODO{Resolver o val_loss que está dando 0 no arquivo. não pode ser val/loss, pq cria uma pasta}
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.best_ckpt_location,
            filename=self.best_ckpt_filename,
            save_top_k=1,
            verbose=True,
            monitor="val|MSE",
            mode='min'
        )
        every_checkpoint_callback = ModelCheckpoint(
            dirpath=self.every_ckpt_location,
            filename=self.every_ckpt_filename,
            save_last=True,
            every_n_epochs=1,
            verbose=True,
            monitor="val|MSE",
            mode='min'
        )

        self.callbacks.append(best_checkpoint_callback)
        self.callbacks.append(every_checkpoint_callback)


        if self.args.early_stopping:
            self.callbacks.append(EarlyStopping(monitor="val|MSE",
                                           patience=self.args.patience))

        if self.resume_ckpt is not None:
            logger_name = f"{self.master_logger_model}_{self.task}_{self.args.dataset}_{self.args.resolution}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}_resume_from_{self.number_last_epoch}"

        else:
            logger_name = f"{self.master_logger_model}_{self.task}_{self.args.dataset}_{self.args.resolution}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}"

        self.logger_master = WandbLogger(project='EnergyConsumption',
                             tags=[self.args.dataset, self.task, self.args.model, self.args.participant_id, self.args.resolution],
                             offline=False,
                             name=logger_name,
                             config=self.args)


        self.logger_slave = CSVLogger(self.local_logger_dir, name = logger_name)
        self.logger_master.log_hyperparams(self.regressor.hparams)
        self.logger_master.watch(self.regressor, log='all')
        self.logger_slave.log_hyperparams(self.regressor.hparams)


        trainer = pl.Trainer(
            logger=[self.logger_master, self.logger_slave],
            enable_checkpointing=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=60,
            resume_from_checkpoint=self.resume_ckpt,
            accumulate_grad_batches=1
        )

        trainer.fit(self.regressor, self.data_module.train_dataloader(), self.data_module.val_dataloader())

