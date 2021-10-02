import pandas as pd
import numpy as np

from src.utils.functions import mkdir_if_not_exists
from tqdm import tqdm
from src.utils.functions import create_sequences
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


class PecanParticipantPreProcessing:
    def __init__(self, individual_id, root_path, sequence_length = 120):
        self.individual_id = individual_id
        self.root_path = root_path
        self.sequence_length = sequence_length

        if Path(f"{self.root_path}/features/{self.individual_id}_features.csv").is_file():
            self.features_df = pd.read_csv(f"{self.root_path}/features/{self.individual_id}_features.csv")
            print(f"[!] - Trainable dataframe shape - {self.features_df.shape}")
        else:
            self.individual_data = pd.read_csv(f'{self.root_path}/{self.individual_id}.csv')
            self.individual_data = self.individual_data.sort_values(by="localminute").reset_index(drop=True)
            print(f"[!] - Shape of initial data: {self.individual_data.shape}")
            self.pre_processing_data()
        self._get_split_data_normalized()

    def _get_split_data_normalized(self):
        n = len(self.features_df)

        self.train_df = self.features_df[0: int(n * .7)]
        self.val_df = self.features_df[int(n * .7): int(n * (1.1 - .2))]
        self.test_df = self.features_df[int(n * (1.0 - .1)):]

        print(f"[*] Train dataframe shape: {self.train_df.shape}")
        print(f"[*] Validation dataframe shape: {self.val_df.shape}")
        print(f"[*] Test dataframe shape: {self.test_df.shape}")

        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.scaler = self.scaler.fit(self.train_df)

        self.train_df = pd.DataFrame(
            self.scaler.transform(self.train_df),
            index=self.train_df.index,
            columns=self.train_df.columns
        )

        self.test_df = pd.DataFrame(
            self.scaler.transform(self.test_df),
            index=self.test_df.index,
            columns=self.test_df.columns
        )
        self.val_df = pd.DataFrame(
            self.scaler.transform(self.val_df),
            index=self.val_df.index,
            columns=self.val_df.columns
        )


        self.train_sequences = create_sequences(self.train_df, 'consumption', self.sequence_length)
        self.test_sequences = create_sequences(self.test_df, 'consumption', self.sequence_length)
        self.val_sequences = create_sequences(self.val_df, 'consumption', self.sequence_length)

        print(f"[!] Train sequence shape: {self.train_sequences[0][0].shape}")
        print(f"[!] Test sequence shape: {self.test_sequences[0][0].shape}")
        print(f"[!] Val sequence shape: {self.val_sequences[0][0].shape}")
        print(f"[!] Len of train, val and test sequence:", len(self.train_sequences), len(self.val_sequences), len(self.test_sequences))

    def get_sequences(self):
        return self.train_sequences, self.test_sequences, self.val_sequences

    def get_n_features(self):
        return self.train_df.shape

    def get_scaler(self):
        return self.scaler

    def get_test_data(self):
        return self.test_df

    def pre_processing_data(self):
        new_data = self.individual_data.copy()

        new_data['generation_solar1'] = np.where(new_data['solar'] < 0, 0, new_data['solar'])
        new_data['generation_solar2'] = np.where(new_data['solar2'] < 0, 0, new_data['solar2'])

        del new_data['dataid'], new_data['solar'], new_data['solar2'], new_data['leg1v'], new_data['leg2v']
        data_columns = list(new_data.columns)

        consumption = data_columns[1:len(data_columns) - 3]
        new_data["sum_consumption"] = new_data[consumption].sum(axis=1)

        generation = data_columns[len(data_columns) - 2:]
        new_data["sum_generation"] = new_data[generation].sum(axis=1)

        compiled = pd.DataFrame({'date': new_data['localminute'], 'consumption': new_data['sum_consumption'],
                                 'generation': new_data['sum_generation']})
        df = compiled.copy()
        rows = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            date_format = pd.Timestamp(row.date)
            row_data = dict(
                consumption=row.consumption,
                generation=row.generation,
                time_hour=date_format.hour,
                time_minute=date_format.minute,
                month=date_format.month,
                day_of_week=date_format.dayofweek,
                day=date_format.day,
                week_of_year=date_format.week
            )
            rows.append(row_data)
        self.features_df = pd.DataFrame(rows)


        date_time = pd.to_datetime(df.pop('date'), format='%Y.%m.%d %H:%M:%S', utc=True)
        timestamp_s = date_time.map(datetime.timestamp)
        day = 24 * (60 ** 2)
        year = (365.2425) * day

        self.features_df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.features_df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        self.features_df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.features_df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        print(f"[!] - Trainable dataframe shape - {self.features_df.shape}")
        print("[!] - Exporting trainable dataframe")

        mkdir_if_not_exists(f"{self.root_path}/features")
        self.features_df.to_csv(f"{self.root_path}/features/{self.individual_id}_features.csv")




