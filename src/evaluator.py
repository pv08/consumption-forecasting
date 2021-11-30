import pytorch_lightning as pl
import torch as T
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import _regressor_eval_class_dict, save_model_figures, save_importances_figure, save_shap_plot, mkdir_if_not_exists

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
        self.regressor._set_model_interpretability_baseline(self.data_module.train_dataloader(), self.args.device)

    def eval(self):
        trainer = pl.Trainer(
            checkpoint_callback=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=30,
            resume_from_checkpoint=self.resume_ckpt
        )

        background_batch = next(iter(self.data_module._shap_background_dataloader()))
        test_batch = next(iter(self.data_module._shap_test_dataloader()))
        T.backends.cudnn.enabled = False
        self._interpretability_background = background_batch['sequence'].to(self.args.device)
        self._interpretability_test = test_batch['sequence'].to(self.args.device)
        explainer = shap.DeepExplainer(self.regressor.model, self._interpretability_background)
        shap_values = explainer.shap_values(self._interpretability_test)
        shap.initjs()
        mkdir_if_not_exists('etc/')
        mkdir_if_not_exists('etc/imgs')
        mkdir_if_not_exists('etc/imgs/features')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots/{self.args.model}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots/{self.args.model}')

        shap_values_2D = shap_values.reshape(-1, 28)
        interpretability_test = self._interpretability_background.reshape(-1, 28)
        interpretability_test_df = pd.DataFrame(data=interpretability_test, columns=self.pecan_dataset.get_features_names())

        shap.summary_plot(shap_values_2D, interpretability_test_df, show=False)
        plt.savefig(
            f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots/{self.args.model}/{self.args.participant_id}_{self.args.model}_bee_summ.svg',
            dpi=600, bbox_inches='tight')
        shap.summary_plot(shap_values_2D, interpretability_test_df, plot_type='bar', show=False)
        plt.savefig(
            f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots/{self.args.model}/{self.args.participant_id}_{self.args.model}_bar_summ.svg',
            dpi=600, bbox_inches='tight')

        shap.force_plot(explainer.expected_value[0], shap_values[0][0], self.pecan_dataset.get_features_names(), show=False)
        plt.savefig(
            f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots/{self.args.model}/{self.args.participant_id}_{self.args.model}_force.svg',
            dpi=600, bbox_inches='tight')

        result = trainer.test(self.regressor, self.data_module.test_dataloader())

        captum_result, shap_result = self.regressor._get_model_interpretability_results()


        save_importances_figure(self.args.participant_id, self.args.model, self.pecan_dataset.get_features_names(),
                                self.regressor._get_feature_importance_index())
        return result


    def predict(self):
        trainer = pl.Trainer(
            checkpoint_callback=True,
            callbacks=self.callbacks,
            max_epochs=self.args.n_epochs,
            gpus=1,
            progress_bar_refresh_rate=30,
            resume_from_checkpoint=self.resume_ckpt,
        )

        result = trainer.predict(self.regressor, self.data_module.test_dataloader())
        save_model_figures(self.args.model, result, self.args.participant_id)

        return None

    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()

    def get_importance_features(self):
        return self.regressor.get_feature_importance_index()
