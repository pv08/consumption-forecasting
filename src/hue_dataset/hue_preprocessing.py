import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from datetime import datetime
from src.utils.functions import mkdir_if_not_exists, create_sequences
from typing import List
from src.pecan_wrapper.basic_dataset import BasicDataset


"""
@data{DVN/N3HGRN_2018,
author = {Makonin, Stephen},
publisher = {Harvard Dataverse},
title = {{HUE: The Hourly Usage of Energy Dataset for Buildings in British Columbia}},
UNF = {UNF:6:F2ursIn9woKDFzaliyA5EA==},
year = {2018},
version = {V5},
doi = {10.7910/DVN/N3HGRN},
url = {https://doi.org/10.7910/DVN/N3HGRN}
}
"""
class HUEPreProcessing(BasicDataset):
    def __init__(self, root_path, id, debug, debug_percent,
                 pca_features = None, sequence_length: int = 60,
                 target_column: str = 'consumption', resolution = '1hour', type='all', shap_model=''):
        super(HUEPreProcessing, self).__init__(root_path=root_path, id=id, sequence_length=sequence_length, target_column=target_column, resolution=resolution, type=type)
        #TODO{colocar o PCA features depois}
        try:
            data = pd.read_csv(f"{self.root_path}/HUE/participants_data/{self.resolution}/features/{self.type}/residential_{str(self.id)}{shap_model}.csv").sort_values(by=['year', 'month', 'day', 'hour'])
            self.usable_data = data.iloc[:int(len(data) * debug_percent)] if debug else data
        except:
            self.generateFile()
            raise FileExistsError("[*] - File not exist. Generated features files. Try again!")
        self.preProcessData(data=self.usable_data)


    def preProcessData(self, data):
        data = data.fillna(0)
        self.features_df = data
        self.n_features = len(self.features_df.columns.to_list())

        n = len(self.features_df)
        self.train_df = self.features_df[0: int(n * .7)]
        self.val_df = self.features_df[int(n * .7): int(n * (1.1 - .2))]
        self.test_df = self.features_df[int(n * (1.0 - .1)):]

        self.scaler = self.scaler.fit(self.features_df)

        self.scalable_train_df = pd.DataFrame(
            self.scaler.transform(self.train_df),
            index=self.train_df.index,
            columns=self.train_df.columns
        )

        self.scalable_test_df = pd.DataFrame(
            self.scaler.transform(self.test_df),
            index=self.test_df.index,
            columns=self.test_df.columns
        )
        self.scalable_val_df = pd.DataFrame(
            self.scaler.transform(self.val_df),
            index=self.val_df.index,
            columns=self.val_df.columns
        )

        self.train_sequences = create_sequences(input_data=self.scalable_train_df,target_column=self.target_column,
                                                sequence_lenght=self.sequence_length)

        self.val_sequences = create_sequences(input_data=self.scalable_val_df,target_column=self.target_column,
                                                sequence_lenght=self.sequence_length)


        self.test_sequences = create_sequences(input_data=self.scalable_test_df,target_column=self.target_column,
                                                sequence_lenght=self.sequence_length)



    def generateFile(self):
        def holiday_search(date, holiday_data):
            result = holiday_data[holiday_data['date'] == date].to_dict('records')
            if len(result) > 0:
                return 1
            else:
                return 0

        def weather_search(date, hour, weather_data):
            result = weather_data[(weather_data['date'] == date) & (weather_data['hour'] == hour)].to_dict('records')
            if len(result) == 1:
                return result[0]
            else:
                return None

        try:
            energy_data = pd.read_csv(f'{self.root_path}/HUE/{self.resolution}/Residential_{str(self.id)}.csv', sep=',').sort_values(by=['date', 'hour'])
            #TODO{a localidade do tempo vai depender do sumário da pasta. FAzer uma função para pegar essa localidade}
            weather_data = pd.read_csv(f'{self.root_path}/HUE/{self.resolution}/Weather_YVR.csv', sep=',', index_col=False).sort_values(
                by=['date', 'hour']).fillna(method='pad')
            holiday_data = pd.read_csv(f'{self.root_path}/HUE/{self.resolution}/Holidays.csv', sep=',')
        except:
            raise FileExistsError("[!] - Make sure that holiday, weather and residential load exists")
        le = LabelEncoder()
        le.fit(list(weather_data.weather.unique()))
        weather_data.weather = le.transform(weather_data.weather)
        energy_data['prev_consumption'] = energy_data.shift(1)['energy_kWh']
        energy_data['consumption_change'] = energy_data.apply(
            lambda row: 0 if np.isnan(row.prev_consumption) else row.energy_kWh - row.prev_consumption, axis=1
        )

        energy_data['year'] = pd.DatetimeIndex(energy_data['date']).year
        energy_data['month'] = pd.DatetimeIndex(energy_data['date']).month
        energy_data['day'] = pd.DatetimeIndex(energy_data['date']).day
        energy_data['dayOfWeek'] = pd.DatetimeIndex(energy_data['date']).day_of_week
        energy_data['dayOfYear'] = pd.DatetimeIndex(energy_data['date']).day_of_year
        complete_row = []
        for _, row in tqdm(energy_data.iterrows(), total=energy_data.shape[0]):
            hour = row.hour
            if row.hour == 0:
                hour = 1
            weather_result = weather_search(row.date, hour, weather_data)
            holiday = holiday_search(row.date, holiday_data)
            if weather_result is not None:
                complete_row.append({
                    'consumption': row.energy_kWh,
                    'prev_consumption': row.prev_consumption,
                    'consumption_change': row.consumption_change,
                    'day': row.day,
                    'month': row.month,
                    'year': row.year,
                    'dayOfWeek': row.dayOfWeek,
                    'dayOfYear': row.dayOfYear,
                    'hour': row.hour,
                    'temperature': weather_result['temperature'],
                    'humidity': weather_result['humidity'],
                    'pressure': weather_result['pressure'],
                    'weather': weather_result['weather'],
                    'holiday': holiday
                })
        df = pd.DataFrame(complete_row)
        date_time = pd.to_datetime(energy_data['date'], format='%Y-%m-%d', utc=True)
        timestamp_s = date_time.map(datetime.timestamp)
        timestamp_s = date_time.map(datetime.timestamp)
        day = 24 * (60 ** 2)
        year = (365.2425) * day

        # df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        # df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        mkdir_if_not_exists(f'{self.root_path}/HUE')
        mkdir_if_not_exists(f'{self.root_path}/HUE/{self.resolution}')
        mkdir_if_not_exists(f'{self.root_path}/HUE/{self.resolution}/features/')
        df = df.fillna(0)
        df.to_csv(f'{self.root_path}/HUE/{self.resolution}/features/{self.type}/residential_{str(self.id)}.csv', index=False)



