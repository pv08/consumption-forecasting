import pytorch_lightning as pl
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import write_test_json

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
        result = trainer.validate(self.regressor, self.data_module.val_dataloader())
        write_test_json(path=self.local_result_dir, result=self.regressor.val_predictions, model=self.args.model, task='validate')

        result = trainer.test(self.regressor, self.data_module.test_dataloader())

        write_test_json(path=self.local_result_dir, result=result, model=self.args.model, task=self.args.task)
        write_test_json(path=self.local_result_dir, result=self.regressor.test_predictions, model=self.args.model, task='predict')


        return result


    def get_len_test_df(self): #test
        return self.pecan_dataset.get_test_data()

    def get_importance_features(self):
        return self.regressor.get_feature_importance_index()
