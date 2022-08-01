import torch as T
import pandas as pd
import numpy as np
from src.utils.functions import create_sequences
from torch.utils.data import DataLoader
from src.dataset import PecanDataset
from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from sklearn.preprocessing import MinMaxScaler

def main():
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    ckpt = 'etc/ckpts/participants/661_test_30_all/sigmoid\LSTM/best/best-LSTM-chpkt-pecanstreet-participant-id-661_test_30_all_epoch=220.ckpt'
    pecan_dataset = PecanParticipantPreProcessing('661_test_30_all', 'data/participants_data/1min',
                                                  60, task='test')
    original_data = pd.read_csv("data/participants_data/1min/features/661_features.csv", sep=',')
    model_features = pd.read_csv("data/participants_data/1min/features/661_test_30_all_features.csv", sep=',')
    original_data = pd.DataFrame(pecan_dataset.scaler.transform(original_data), index=original_data.index,
                                 columns=original_data.columns
    )
    test_df = pecan_dataset.test_df.copy()
    forecasting_data = original_data.iloc[129086:129350] # estimativa de dados para os próximos 15 minutos
    initial_predicition = test_df.iloc[-61:].copy()
    initial_sequence = create_sequences(initial_predicition, 'consumption', 60)
    forecasting_dataset_sequence = PecanDataset(initial_sequence, device)

    forecasting_module = DataLoader(dataset=forecasting_dataset_sequence,
                                                 batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    model = ConsumptionRNNRegressor.load_from_checkpoint(checkpoint_path=ckpt,
                                                         strict=False,
                                                         scaler=pecan_dataset.scaler,
                                                         device=device,
                                                         n_features=pecan_dataset.get_n_features(),
                                                         lr=1e-5,
                                                         n_hidden=256,
                                                         n_layers=3,
                                                         dropout=0.3,
                                                         activation_function='sigmoid')


    model.freeze()
    forecasting_starting_point = 129086
    new_scaler = MinMaxScaler()
    new_scaler.min_, new_scaler.scale_ = pecan_dataset.scaler.min_[0], pecan_dataset.scaler.scale_[0]
    for i in range(15): #prever os próximos 15 minutos
        batch = next(iter(forecasting_module))
        _, output = model(batch['sequence'].to(device))
        forecasting_data.at[forecasting_starting_point, 'consumption'] = output.item()
        test_df.loc[forecasting_starting_point] = forecasting_data.iloc[i]
        next_prediction = test_df.iloc[-61:].copy()
        next_sequence = create_sequences(next_prediction, 'consumption', 60)

        next_dataset_sequence = PecanDataset(next_sequence, device)

        forecasting_module = DataLoader(dataset=next_dataset_sequence,
                                                 batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

        forecasting_starting_point += 1
    test_df

if __name__ == '__main__':
    main()