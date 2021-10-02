from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.dataset import PecanDataset, PecanDataModule
from src.utils.functions import mkdir_if_not_exists

class PecanWrapper:
    def __init__(self, args):
        self.args = args
        self.callbacks = []
        self.pecan_dataset = PecanParticipantPreProcessing(self.args.participant_id, self.args.root_path,
                                                           self.args.sequence_length)

        self.train_sequences, self.test_sequences, self.val_sequences = self.pecan_dataset.get_sequences()
        self.args.n_features = self.pecan_dataset.get_n_features()[1]

        self.data_module = PecanDataModule(
            self.args.device,
            self.train_sequences,
            self.test_sequences,
            self.val_sequences,
            self.args.batch_size,
            self.args.num_workers
        )
        self.data_module.setup()

        print(f'[!] - Training shape:', end='\n')
        train_dataset = PecanDataset(self.train_sequences, self.args.device)
        for item in train_dataset:
            print(f"[*] - Sequence shape: {item['sequence'].shape}")
            print(f"[*] - Lables shape: {item['label'].shape}")
            print(item['label'])
            break
        mkdir_if_not_exists('checkpoints/')
        mkdir_if_not_exists('checkpoints/participants/')
        mkdir_if_not_exists(f'checkpoints/participants/{self.args.participant_id}/')
        mkdir_if_not_exists(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/')
        mkdir_if_not_exists(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/')
        mkdir_if_not_exists(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/best/')
        mkdir_if_not_exists(f'checkpoints/participants/{self.args.participant_id}/{self.args.activation_fn}/{self.args.model}/epochs/')







