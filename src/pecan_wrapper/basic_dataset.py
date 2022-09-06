from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Any
class BasicDataset:
    def __init__(self, root_path, id, sequence_length, target_column, resolution):
        self.root_path = root_path
        self.id = id
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.resolution = resolution
        self.n_features = 0
        self.train_sequences = None
        self.val_sequences = None
        self.test_sequences = None

    def getNFeatures(self):
        return self.n_features

    def preProcessData(self, data):
        raise NotImplementedError


