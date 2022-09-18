import pandas as pd
import numpy as np
from src.utils.functions import mkdir_if_not_exists, mk_weather_data, create_sequences
from tqdm import tqdm

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from src.pecan_wrapper.basic_dataset import BasicDataset

class PecanParticipantPreProcessing(BasicDataset):
    def __init__(self, root_path, id, sequence_length = 120, shap_sequence = 30,
                 target_column = 'consumption', task = 'train', resolution = '1min', type='all', shap_model = ''):
        super(PecanParticipantPreProcessing, self).__init__(root_path=root_path, id=id, sequence_length=sequence_length, target_column=target_column, resolution=resolution, type=type)
        self.task = task
        self.key = '53a4996903bc42d9a47162143210210'  # API key obtained from https://www.worldweatheronline.com/
        self.locations = [
            '162.89.0.47']  # list of strings containg US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude (decimal degree) or city name
        self.start = '01-01-2018'  # date when desired scraping period starts; preferred date format: 'dd-mmm-yyyy'
        self.end = '31-12-2018'  # date when desired scraping period ends; preferred date format: 'dd-mmm-yyyy
        self.freq = 1  # frequency between observations; possible values 1 (1 hour), 3 (3 hours), 6 (6 hours), 12 (12 hours (day/night)) or 24 (daily averages)weather_df = mk_weather_data()
        self.resolution = resolution
        self._pecan_idx_column = {
            '1min': 'localminute',
            '15min': 'local_15min'
        }


        if Path(f"{self.root_path}/Pecanstreet/participants_data/{resolution}/features/{self.type}/{self.id}_{self._data_type[self.type]}{shap_model}.csv").is_file():
            self.features_df = pd.read_csv(f"{self.root_path}/Pecanstreet/participants_data/{resolution}/features/{self.type}/{self.id}_{self._data_type[self.type]}.csv")
            print(f"[!] - Trainable dataframe shape - {self.features_df.shape}")
        else:
            self.individual_data = pd.read_csv(f'{self.root_path}/Pecanstreet/participants_data/{resolution}/{self.id}.csv')
            self.individual_data = self.individual_data.sort_values(by=self._pecan_idx_column[resolution]).reset_index(drop=True)
            print(f"[!] - Shape of initial data: {self.individual_data.shape}")
            self.pre_processing_data()
        self.preProcessData(self.features_df)

    def preProcessData(self, df):
        n = len(df)

        self.n_features = len(df.columns.to_list())

        self.train_df = df[0: int(n * .7)]
        self.val_df = df[int(n * .7): int(n * (1.1 - .2))]
        self.test_df = df[int(n * (1.0 - .1)):]

        print(f"[*] Train dataframe shape: {self.train_df.shape}")
        print(f"[*] Validation dataframe shape: {self.val_df.shape}")
        print(f"[*] Test dataframe shape: {self.test_df.shape}")

        self.scaler = self.scaler.fit(df)

        self.total_df = pd.DataFrame(
            self.scaler.transform(df),
            index=df.index,
            columns=df.columns
        )

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


        # self.shap_background_sequence = create_sequences(self.total_df[:int(len(self.total_df)*.5)], 'consumption', self.shap_sequence)
        # self.shap_test_sequence = create_sequences(self.total_df[:int(len(self.total_df)*.5)], 'consumption', self.shap_sequence)
        #

        self.train_sequences = create_sequences(self.train_df, 'consumption', self.sequence_length)
        self.val_sequences = create_sequences(self.val_df, 'consumption', self.sequence_length)
        self.test_sequences = create_sequences(self.test_df, 'consumption', self.sequence_length)
        print(f"[!] Train sequence shape: {self.train_sequences[0][0].shape}")
        print(f"[!] Val sequence shape: {self.val_sequences[0][0].shape}")
        print(f"[!] Test sequence shape: {self.test_sequences[0][0].shape}")


    def get_sequences(self):
        return self.train_sequences, self.test_sequences, self.val_sequences

    def get_standard_df_features(self):
        return self.total_df

    def get_features_names(self):
        return self.features_df.columns

    def get_n_features(self):
        return self.features_df.shape[1]

    def get_scaler(self):
        return self.scaler

    def get_test_data(self):
        return self.test_df

    def get_features_df(self):
        return self.features_df

    def insert_weather_data(self, date, hour):
        values = {}
        loc = self.weather_df.loc[(self.weather_df['date'] == str(date)) & (self.weather_df['hour'] == f'{str(hour)}:00')]
        for _, row in loc.iterrows():
            for columns in loc.columns[2:-1]:
                values[columns] = row[columns]
        return values

    def pre_processing_data(self):
        if Path(f"{self.root_path}/Pecanstreet/weather_data/162.89.0.47.csv").is_file():
            self.weather_df = pd.read_csv(f"{self.root_path}/Pecanstreet/weather_data/162.89.0.47.csv")
        else:
            self.weather_df = mk_weather_data(self.key, self.locations, self.start, self.end, self.freq)

        self.weather_df['date'] = pd.to_datetime(self.weather_df['date_time'])
        del self.weather_df['moonrise'], self.weather_df['moonset'], self.weather_df['sunrise'], self.weather_df['sunset']

        weather = []
        for _, row in tqdm(self.weather_df.iterrows(), total=self.weather_df.shape[0]):
            values = {
                'date': datetime.strftime(row.date, '%Y-%m-%d'),
                'hour': datetime.strftime(row.date, '%H:%M')
            }
            for columns in self.weather_df.columns[1:-1]:
                values[columns] = row[columns]
            weather.append(values)

        self.weather_df = pd.DataFrame(weather)

        new_data = self.individual_data.copy()
        new_data['crop_date'] = pd.to_datetime(new_data[self._pecan_idx_column[self.resolution]])
        new_data['generation_solar1'] = np.where(new_data['solar'] < 0, 0, new_data['solar'])
        new_data['generation_solar2'] = np.where(new_data['solar2'] < 0, 0, new_data['solar2'])

        del new_data['dataid'], new_data['solar'], new_data['solar2'], new_data['leg1v'], new_data['leg2v']
        data_columns = list(new_data.columns)

        consumption = data_columns[1:len(data_columns) - 3]
        new_data["sum_consumption"] = new_data[consumption].sum(axis=1)

        generation = data_columns[len(data_columns) - 2:]
        new_data["sum_generation"] = new_data[generation].sum(axis=1)

        compiled = pd.DataFrame({'date': new_data[self._pecan_idx_column[self.resolution]], 'consumption': new_data['sum_consumption'],
                                 'generation': new_data['sum_generation'], 'crop_date': new_data['crop_date']})
        df = compiled.copy()
        df['prev_consumption'] = df.shift(1)['consumption']
        df['consumption_change'] = df.apply(
            lambda row: 0 if np.isnan(row.prev_consumption) else row.consumption - row.prev_consumption, axis=1
        )
        rows = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            date_format = pd.Timestamp(row.date)
            row_data = dict(
                date=datetime.strftime(row.crop_date, '%Y-%m-%d'),
                hour=datetime.strftime(row.crop_date, '%H:%M'),
                generation=row.generation,
                time_hour=date_format.hour,
                time_minute=date_format.minute,
                month=date_format.month,
                day_of_week=date_format.dayofweek,
                day=date_format.day,
                week_of_year=date_format.week,
                consumption_change=row.consumption_change,
                consumption=row.consumption,
            )
            weather_data = self.insert_weather_data(datetime.strftime(row.crop_date, '%Y-%m-%d'),
                                               datetime.strftime(row.crop_date, '%H'))
            row_data.update(weather_data)
            rows.append(row_data)
        self.features_df = pd.DataFrame(rows)


        date_time = pd.to_datetime(df.pop('date'), format='%Y.%m.%d %H:%M:%S', utc=True)

        # timestamp_s = date_time.map(datetime.timestamp)
        # day = 24 * (60 ** 2)
        # year = (365.2425) * day
        #
        # self.features_df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        # self.features_df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        #
        # self.features_df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        # self.features_df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        print(f"[!] - Trainable dataframe shape - {self.features_df.shape}")
        print("[!] - Exporting trainable dataframe")

        mkdir_if_not_exists(f"{self.root_path}/Pecanstreet/participants_data/{self.resolution}")
        mkdir_if_not_exists(f"{self.root_path}/Pecanstreet/participants_data/{self.resolution}/features")
        mkdir_if_not_exists(f"{self.root_path}/Pecanstreet/participants_data/{self.resolution}/features/{self.type}")
        del self.features_df['date'], self.features_df['hour']
        self.features_df.to_csv(f"{self.root_path}//Pecanstreet/participants_data/{self.resolution}/features/{self.type}/{self.id}_{self._data_type[type]}.csv", index=False)



