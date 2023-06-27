import pytorch_lightning as pl
import pandas as pd
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from sklearn.preprocessing import MinMaxScaler
from src.utils.functions import write_test_json, save_json_metrics, descale


class PecanEvaluator(PecanWrapper):
    def __init__(self, args):
        super(PecanEvaluator, self).__init__(args)

        self.resume_ckpt, self.number_last_epoch = self.get_epoch_trained(self.best_ckpt_location)


        self.regressor = self._get_trained_regressor_model(self.args, self.resume_ckpt, self.args.scaler)


    def evaluate(self):
        trainer = pl.Trainer(
            gpus=self.args.gpu,
            progress_bar_refresh_rate=60
        )
        validation_result = trainer.validate(self.regressor, self.data_module.val_dataloader())
        validation_result[0]["model"] = self.args.model
        
        for preds in self.regressor.val_predictions:
            preds['model'] = self.args.model
        validation_df = pd.DataFrame(self.regressor.val_predictions)
        validation_df.to_csv(f"{self.local_result_dir}/{self.args.model}/validation_preds.csv")
        save_json_metrics(content=validation_result, path=f"{self.local_result_dir}/", filename=f"validation_metrics_report{self.args.data_type if self.args.resolution == '1min' else ''}", model=self.args.model)
        

        test_result = trainer.test(self.regressor, self.data_module.test_dataloader())
        test_result[0]["model"] = self.args.model

        descaler = MinMaxScaler(feature_range=(1,1))
        scalerlabel_idx = list(self.dataset.scaler.feature_names_in_).index('consumption')
        descaler.min_ = self.dataset.scaler.min_[scalerlabel_idx]
        descaler.scale_ = self.dataset.scaler.scale_[scalerlabel_idx]

        for preds in self.regressor.test_predictions:
            preds['model'] = self.args.model
        test_df = pd.DataFrame(self.regressor.test_predictions)
        test_df.to_csv(f"{self.local_result_dir}/{self.args.model}/escaled_test_preds.csv")
        test_df.label = descale(descaler, test_df.label)
        test_df.model_output = descale(descaler, test_df.model_output)
        test_df.to_csv(f"{self.local_result_dir}/{self.args.model}/test_preds.csv")
        save_json_metrics(content=test_result, path=self.local_result_dir, filename=f"metrics_report", model=self.args.model)

        return test_result


    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()

    def get_importance_features(self):
        return self.regressor.get_feature_importance_index()
