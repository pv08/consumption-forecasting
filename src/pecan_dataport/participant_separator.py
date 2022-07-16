import pandas as pd
from src.utils.functions import get_files_inpath, mkdir_if_not_exists, verify_existence_data
from tqdm import tqdm

class DataportDataDivisor:
    def __init__(self, pecan_path, resolution):
        """
        This class is refered to the manipulation dataset construction.

        :param pecan_path: Base Pecan Street Dataport's dataset path in string format
        :param resolution: format of time: 1s, 1min, 1h
        """

        self.resolution = resolution
        self._ids = []
        self.pecan_path = pecan_path
        self.pecan_files = get_files_inpath(self.pecan_path, '.csv')

        if not len(self.pecan_files) > 0 or len(self.pecan_files) == 0:
            raise Exception("[!] - Files do not found")
        elif len(self.pecan_files) == 1:
            self.pecan_data = pd.read_csv(f"{self.pecan_path}/{self.pecan_files[0]}")
            self.separate_customer_data()
        elif len(self.pecan_files) > 1:
            li = []
            for files in self.pecan_files:
                df = pd.read_csv(f"{self.pecan_path}/{files}")
                self.li.append(df)
            self.pecan_data = pd.concat(li, ignore_index=True)
            self.separate_customer_data()


    def get_ids(self) -> list:
        """

        :return: list of participant's ids
        """
        return self._ids


    def separate_customer_data(self):

        """ Separate Pecan Street load data into. This method separate the loads for each participants,
        saving on data/participants folder """
        self._ids = self.pecan_data['dataid'].unique()
        mkdir_if_not_exists('data/participants_data')
        if self.resolution == '1min':
            mkdir_if_not_exists('data/participants_data/1min')
            self.exportCustomerData('data/participants_data/1min')
        elif self.resolution == '1s':
            mkdir_if_not_exists('data/participants_data/1s')
            self.exportCustomerData('data/participants_data/1s')

    def exportCustomerData(self, path: str):
        """
        Function to create the dataset of participant
        :param path: path to save
        :return: None
        """
        for i in tqdm(range(len(self._ids))):
            customer_data = self.pecan_data.loc[self.pecan_data['dataid'] == self._ids[i]]
            if not verify_existence_data(f'{path}/{self._ids[i]}.csv'):
                customer_data.to_csv(f'{path}/{self._ids[i]}.csv', index=False)

