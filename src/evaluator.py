import pytorch_lightning as pl
import os
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import _regressor_eval_class_dict, save_model_figures
from src.dataset import PecanDataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.loggers import WandbLogger

class PecanEvaluator(PecanWrapper):
    def __init__(self, args):
        super(PecanEvaluator, self).__init__(args)
        self.predictions = []
        self.labels = []

        list_epochs = next(os.walk(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/best/'))[2]
        if len(list_epochs) > 0:
            last_epoch = list_epochs[len(list_epochs) - 1]
            number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
            print(f"[!] - Best Epoch - {number_last_epoch}")

            self.resume_ckpt = f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/best/{last_epoch}'

        else:
            raise SystemError

        self.regressor = _regressor_eval_class_dict(self.args, self.resume_ckpt, self.pecan_dataset.get_scaler())

    def eval(self):


        trainer = pl.Trainer(
            checkpoint_callback=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=30,
            resume_from_checkpoint=self.resume_ckpt
        )

        result = trainer.test(self.regressor, self.data_module.test_dataloader())
        return result

    def predict(self):
        # test_dataset = PecanDataset(self.test_sequences, self.args.device)
        # for item in tqdm(test_dataset):
        #     sequence = item['sequence'].to(self.args.device)
        #     label = item['label'].to(self.args.device)
        #
        #     _, output = self.regressor(sequence.unsqueeze(dim=0))
        #     self.predictions.append(output.item())
        #     self.labels.append(label.item())
        #
        # scaler = self.pecan_dataset.get_scaler()
        # descaler = MinMaxScaler()
        # descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
        #
        # descaled_predictions = descale(descaler, self.predictions)
        # descaled_labels = descale(descaler, self.labels)
        #
        # return descaled_predictions, descaled_labels

        trainer = pl.Trainer(
            checkpoint_callback=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=30,
            resume_from_checkpoint=self.resume_ckpt
        )

        result = trainer.predict(self.regressor, self.data_module.test_dataloader())
        save_model_figures(self.args.model, result)
        return None

    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()




