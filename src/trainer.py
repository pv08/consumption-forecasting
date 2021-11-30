from itertools import accumulate

import pytorch_lightning as pl
import os
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import mkdir_if_not_exists, _regressor_trainer_class_dict


class PecanTrainer(PecanWrapper):
    def __init__(self, args):
        super(PecanTrainer, self).__init__(args)
        self.regressor = _regressor_trainer_class_dict(self.args)

        list_epochs = next(os.walk(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/epochs/'))[2]
        if len(list_epochs) > 0:
            last_epoch = list_epochs[len(list_epochs) - 1]
            number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
            print(f"[!] - Last Epoch - {number_last_epoch}")

            resume_ckpt = f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/epochs/{last_epoch}'
            logger_name = f"PL_{self.args.model}_{self.args.activation_fn}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}_{self.args.lr}_resume_from_{number_last_epoch}"

        else:
            resume_ckpt = None
            logger_name = f"PL_{self.args.model}_{self.args.activation_fn}_{self.args.participant_id}_" \
                          f"{self.args.n_epochs}_{self.args.lr}"

        #TODO{Resolver o val_loss que está dando 0 no arquivo. não pode ser val/loss, pq cria uma pasta}
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/best',
            filename=f'best-{self.args.model}-chpkt-pecanstreet-participant-id-{self.args.participant_id}' + "_{epoch:03d}-{val_loss:.5f}",
            save_top_k=1,
            verbose=True,
            monitor="val/loss_epoch",
            mode='min',
            save_weights_only = True
        )
        every_checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/epochs/',
            filename= str(self.args.model) + '-chpkt-pecanstreet-participant-id-'
                     + str(self.args.participant_id) + '-{epoch:03d}-{val_loss:.5f}',
            save_top_k=-1,
            every_n_epochs=1,
            verbose=True,
            monitor="val/loss_epoch",
            mode='min',
            save_weights_only=True
        )

        self.callbacks.append(best_checkpoint_callback)
        self.callbacks.append(every_checkpoint_callback)


        if self.args.early_stopping:
            self.callbacks.append(EarlyStopping(monitor="val/loss_epoch",
                                           patience=self.args.patience))


        logger = WandbLogger(project='pecanstreet',
                             tags=f"{self.args.model}_regressor_trainer",
                             offline=False,
                             name=logger_name,
                             config=self.args)



        logger.log_hyperparams(self.regressor.hparams)

        logger.watch(self.regressor, log='all')
        self.trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=30,
            resume_from_checkpoint=resume_ckpt,
            accumulate_grad_batches=1
        )

#        self.regressor.set_gradiant_shap_baseline(self.gradient_baseline, self.gradient_baseline_test)

    def train(self):

        for item in self.data_module.train_dataloader():
            print(item['sequence'].shape)
            print(item['label'].shape)
            break

        self.trainer.fit(self.regressor, self.data_module)
