import pytorch_lightning as pl
import os
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import _regressor_eval_class_dict, save_model_figures, mkdir_if_not_exists, write_test_json

class PecanEvaluator(PecanWrapper):
    def __init__(self, args):
        super(PecanEvaluator, self).__init__(args)
        self.regressor = _regressor_eval_class_dict(self.args, self.resume_ckpt, self.pecan_dataset.get_scaler())

        mkdir_if_not_exists('etc/')
        mkdir_if_not_exists('etc/imgs')
        mkdir_if_not_exists('etc/imgs/features')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots/{self.args.model}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots/{self.args.model}')


    def eval(self):
        trainer = pl.Trainer(
            enable_checkpointing=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=60,
            resume_from_checkpoint=self.resume_ckpt
        )

        result = trainer.test(self.regressor, self.data_module.test_dataloader())
        write_test_json(result, self.args.model, self.args.task, self.args.participant_id)
        write_test_json(self.regressor.predictions, self.args.model, 'predict', self.args.participant_id)

        return result


    def predict(self):
        trainer = pl.Trainer(
            enable_checkpointing=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=60,
            resume_from_checkpoint=self.resume_ckpt
        )

        result = trainer.predict(self.regressor, self.data_module.test_dataloader())
        write_test_json(result, self.args.model, self.args.task, self.args.participant_id)
        save_model_figures(self.args.model, result, self.args.participant_id)
        return result

    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()

    def get_importance_features(self):
        return self.regressor.get_feature_importance_index()
