import pytorch_lightning as pl
from src.regressors.linear_regression import ConsumptionLinearRegressor, ConsumptionMLPRegressor
from src.regressors.lstm_regressor import ConsumptionLSTMRegressor
from src.regressors.gru_regression import ConsumptionGRURegressor
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from src.regressors.conv_rnn_regressor import ConsumptionConvRNNRegressor
from src.regressors.transformer_regressor import ConsumptionTransformerRegressor, ConsumptionTSTRegressor
from src.regressors.fcn_regressor import ConsumptionFCNRegressor
from src.regressors.tcn_regressor import ConsumptionTCNRegressor
from src.regressors.resnet_regressor import ConsumptionResNetRegressor
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import mkdir_if_not_exists, write_validation_json

class PecanValidator(PecanWrapper):
    def __init__(self, args):
        super(PecanValidator, self).__init__(args)

        self.resume_ckpt, self.number_last_epoch = self.get_best_epoch_trained(self.args.participant_id,
                                                                               self.args.activation_fn, self.args.model)


        self.regressor = self._get_trained_regressor_model(self.args, self.resume_ckpt, self.args.scaler)

        mkdir_if_not_exists('etc/')
        mkdir_if_not_exists('etc/imgs')
        mkdir_if_not_exists('etc/imgs/features')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/summary_plots/{self.args.model}')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots')
        mkdir_if_not_exists(f'etc/imgs/features/{self.args.participant_id}/shap_values/force_plots/{self.args.model}')

    def validator(self):
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=60
        )
        result = trainer.validate(self.regressor, self.data_module.val_dataloader())
        write_validation_json(self.regressor.predictions, self.args.model, 'predict', self.args.participant_id)

        return result


    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()

    def get_importance_features(self):
        return self.regressor.get_feature_importance_index()
