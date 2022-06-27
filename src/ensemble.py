import pytorch_lightning as pl
from src.pecan_wrapper.basic_wrapper import PecanWrapper
from src.utils.functions import mkdir_if_not_exists, _regressor_ensemble_class_dict, write_test_json

class PecanEnsemble(PecanWrapper):
    def __init__(self, args):
        super(PecanEnsemble, self).__init__(args)
        self.args = args

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


        self.regressor = _regressor_ensemble_class_dict(self.args)

    def ensemble(self):
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=60
        )

        result = trainer.test(self.regressor, self.data_module.test_dataloader())
        write_test_json(result, self.ensemble_name, 'test', self.args.participant_id)
        write_test_json(self.regressor.predictions, self.ensemble_name, 'predict', self.args.participant_id)



