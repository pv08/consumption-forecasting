import os
import numpy as np
import wwo_hist
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.regressors.linear_regression import ConsumptionLinearRegressor, ConsumptionMLPRegressor
from src.regressors.lstm_regressor import ConsumptionLSTMRegressor
from src.regressors.gru_regression import ConsumptionGRURegressor
from src.regressors.rnn_regressor import ConsumptionRNNRegressor
from src.regressors.conv_rnn_regressor import ConsumptionConvRNNRegressor
from src.regressors.transformer_regressor import ConsumptionTransformerRegressor, ConsumptionTSTRegressor
from src.regressors.fcn_regressor import ConsumptionFCNRegressor
from src.regressors.tcn_regressor import ConsumptionTCNRegressor
from src.regressors.resnet_regressor import ConsumptionResNetRegressor
from tqdm import tqdm


def mk_weather_data(api, location, start, end, freq):
    mkdir_if_not_exists("data/weather_data/")
    os.chdir("data/weather_data/")

    hist_weather_data = wwo_hist.retrieve_hist_data(
        api_key=api,
        location_list=location,
        start_date=start,
        end_date=end,
        frequency=freq,
        location_label=False,
        export_csv=True,
        store_df=False
    )
    os.chdir("../../")
    return pd.read_csv(f'data/weather_data/{location[0]}.csv')

def save_importances_figure(participant, model, features, importances: dict, x_axis_title = 'Features', y_axis_title = 'Degree'):

    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/imgs/')
    mkdir_if_not_exists('etc/imgs/features')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/{model}')

    x_pos = (np.arange(len(features)))
    plt.figure(figsize=(35, 20))
    plt.title(f'[`{model}`] - Avg. Feature Importance', fontsize=15)
    plt.plot(x_pos, np.mean(importances['IntegratedGradients'], axis=0), 'o-', label='IntegratedGradients',
             color='blue')
    plt.xticks(x_pos, features, wrap=True)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.legend(importances.keys(), fontsize=15)
    plt.savefig(f'etc/imgs/features/{participant}_{model}_model_feature_importances.svg', dpi=600, bbox_inches='tight')


def save_pca_features(pca_variance):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/imgs')
    mkdir_if_not_exists('etc/imgs/PCA')

    plt.figure(figsize=(20, 20))
    plt.title(f'[`PCA`] - Features componentes', fontsize=15)
    plt.plot(np.cumsum(pca_variance))

    plt.xlabel('Number Componentes', fontsize=15)
    plt.ylabel('Cumulative Explained Variance', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(f'etc/imgs/PCA/features_variace.png', dpi=600, bbox_inches='tight')


def save_shap_plot(participant, explainer, shap_values, interpretability_test,  model_name, features):
    shap.initjs()
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/imgs')
    mkdir_if_not_exists('etc/imgs/features')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/shap_values')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/shap_values/summary_plots')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/shap_values/summary_plots/{model_name}')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/shap_values/force_plots')
    mkdir_if_not_exists(f'etc/imgs/features/{participant}/shap_values/force_plots/{model_name}')






def save_model_figures(model, results, participant):
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/imgs')
    mkdir_if_not_exists('etc/imgs/predictions')
    mkdir_if_not_exists(f'etc/imgs/predictions/{participant}')
    mkdir_if_not_exists(f'etc/imgs/predictions/{participant}/{model}')

    output = [result[0] for result in results]
    label = [result[1] for result in results]

    plt.figure(figsize=(20, 10))
    plt.title(f'[`{model}`] - Relationship between prediction and real data', fontsize=15)
    plt.plot(label, 'c-', label='real')
    plt.plot(output, 'm--', label='predictions')

    plt.xlabel('time [`min`]', fontsize=15)
    plt.ylabel('consumption [`kw`]', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(f'etc/imgs/predictions/{participant}_{model}_model.svg', dpi=600, bbox_inches='tight')


def _regressor_trainer_class_dict(args):
    returned_model = {
        "LSTM": ConsumptionLSTMRegressor(device=args.device,
                                         n_features=args.n_features,
                                         lr=args.lr,
                                         n_hidden=args.n_hidden,
                                         n_layers=args.n_layers,
                                         dropout=args.dropout,
                                         activation_function=args.activation_fn,
                                         bidirectional=args.bidirectional),
        "RNN": ConsumptionRNNRegressor(device=args.device,
                                         n_features=args.n_features,
                                         lr=args.lr,
                                         n_hidden=args.n_hidden,
                                         n_layers=args.n_layers,
                                         dropout=args.dropout,
                                         activation_function=args.activation_fn),
        "GRU": ConsumptionGRURegressor(device=args.device,
                                       n_features=args.n_features,
                                       lr=args.lr,
                                       n_hidden=args.n_hidden,
                                       n_layers=args.n_layers,
                                       dropout=args.dropout,
                                       activation_function=args.activation_fn),
        "Linear": ConsumptionLinearRegressor(device=args.device,
                                           n_features=args.n_features,
                                           lr=args.lr,
                                           n_hidden=args.n_hidden,
                                           activation_function=args.activation_fn),
        "MLP": ConsumptionMLPRegressor(device=args.device,
                                           n_features=args.n_features,
                                            sequence_length=args.sequence_length,
                                           lr=args.lr),
        "FCN": ConsumptionFCNRegressor(device=args.device,
                                           n_features=args.n_features,
                                           lr=args.lr, activation_function=args.activation_fn),
        "TCN": ConsumptionTCNRegressor(device=args.device,
                                           n_features=args.n_features,
                                           lr=args.lr, activation_function=args.activation_fn),
        "ResNet": ConsumptionResNetRegressor(device=args.device,
                                           n_features=args.n_features,
                                           lr=args.lr, activation_function=args.activation_fn),
        "ConvRNN": ConsumptionConvRNNRegressor(device=args.device,
                                            n_features=args.n_features,
                                            time_steps=args.sequence_length,
                                            lr=args.lr, activation_function=args.activation_fn),
        "Transformer": ConsumptionTransformerRegressor(device=args.device,
                                                n_features=args.n_features,
                                                d_model=args.d_model,
                                                n_head=args.n_head,
                                                d_ffn=args.d_ffn,
                                                dropout=args.dropout,
                                                n_layers=args.n_layers,
                                                lr=args.lr,
                                                activation_function=args.tst_activation_fn),
        'TST': ConsumptionTSTRegressor(device=args.device, n_features=args.n_features, seq_len=args.sequence_length,
                                        max_seq_len=args.max_seq_len, d_model=args.d_model, n_head=args.n_head,
                                        d_k=args.d_k, d_v=args.d_v, d_ffn=args.d_ffn, res_dropout=args.res_dropout,
                                        n_layers=args.n_layers, lr=args.lr,  activation_function=args.tst_activation_fn,
                                        fc_dropout=args.fc_dropout)
    }
    try:
        return returned_model[args.model]
    except:
        raise NotImplementedError(f"[?] - Model not implemented yet")


def _regressor_eval_class_dict(args, ckpt, scaler):
    if args.model == 'LSTM':
        model = ConsumptionLSTMRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                         n_features=args.n_features,
                                         lr=args.lr,
                                         n_hidden=args.n_hidden,
                                         n_layers=args.n_layers,
                                         dropout=args.dropout,
                                         activation_function=args.activation_fn,
                                         bidirectional=args.bidirectional, scaler=scaler)
    elif args.model == 'RNN':
        model = ConsumptionRNNRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                         n_features=args.n_features,
                                         lr=args.lr,
                                         n_hidden=args.n_hidden,
                                         n_layers=args.n_layers,
                                         dropout=args.dropout,
                                         activation_function=args.activation_fn, scaler=scaler)
    elif args.model == 'GRU':
        model = ConsumptionGRURegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                       n_features=args.n_features,
                                       lr=args.lr,
                                       n_hidden=args.n_hidden,
                                       n_layers=args.n_layers,
                                       dropout=args.dropout,
                                       activation_function=args.activation_fn, scaler=scaler)
    elif args.model == 'Linear':
        model = ConsumptionLinearRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                             n_features=args.n_features,
                                             lr=args.lr,
                                             n_hidden=args.n_hidden,
                                             activation_function=args.activation_fn, scaler=scaler)
    elif args.model == 'Transformer':
        model = ConsumptionTransformerRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False,
                                                             device=args.device,
                                                             n_features=args.n_features,
                                                             d_model=args.d_model,
                                                             n_head=args.n_head,
                                                             d_ffn=args.d_ffn,
                                                             dropout=args.dropout,
                                                             n_layers=args.n_layers,
                                                             lr=args.lr,
                                                             activation_function=args.tst_activation_fn,
                                                             scaler=scaler)
    elif args.model == 'TST':
        model = ConsumptionTSTRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                                             n_features=args.n_features, seq_len=args.sequence_length,
                                        max_seq_len=args.max_seq_len, d_model=args.d_model, n_head=args.n_head,
                                        d_k=args.d_k, d_v=args.d_v, d_ffn=args.d_ffn, res_dropout=args.res_dropout,
                                        n_layers=args.n_layers, lr=args.lr,  activation_function=args.tst_activation_fn,
                                        fc_dropout=args.fc_dropout, scaler=scaler)
    elif args.model == 'ConvRNN':
        model = ConsumptionConvRNNRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                            n_features=args.n_features,
                                            time_steps=args.sequence_length,
                                            lr=args.lr, activation_function=args.activation_fn, scaler=scaler)
    elif args.model == 'ResNet':
        model = ConsumptionResNetRegressor.load_from_checkpoint(checkpoint_path=ckpt, strict=False, device=args.device,
                                           n_features=args.n_features,
                                           lr=args.lr, activation_function=args.activation_fn,scaler=scaler)
    else:
        raise NotImplementedError(f"[?] - Model not implemented yet")

    return model


def _get_resume_and_best_epoch(participant, model):
    list_epochs = next(os.walk(f'checkpoints/participants/{participant}/{model}/train/epochs/'))[2]
    if len(list_epochs) > 0:
        last_epoch = list_epochs[len(list_epochs) - 1]
        number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]

        train_resume_ckpt = f'checkpoints/participants/{participant}/{model}/train/epochs/{number_last_epoch}'
    else:
        train_resume_ckpt = None


    list_best_epochs = next(os.walk(f'checkpoints/participants/{participant}/{model}/train/best/'))[2]
    if len(list_best_epochs) > 0:
        last_epoch = list_best_epochs[len(list_best_epochs) - 1]
        number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
        print(f"[!] - Best Epoch - {number_last_epoch}")

        eval_best_epoch = f'checkpoints/participants/{participant}/{model}/train/best/{last_epoch}'
    else:
        eval_best_epoch = None

    return train_resume_ckpt, eval_best_epoch


def create_sequences(input_data:pd.DataFrame, target_column, sequence_lenght):

    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_lenght)):
        sequence = input_data[i:i+sequence_lenght]
        label_position = i + sequence_lenght
        label = input_data.iloc[label_position][target_column]
        # del sequence[target_column]
        sequences.append((sequence, label))

    return sequences


def get_files_inpath(path: str, extension: str) -> list:
    """
    Verify the extension files inside the path and return a list of files
    :param path: str() -> path to verify
    :param extension: str() -> .csv for example
    :return: list() -> list of files with the extension
    """
    filenames = os.listdir(path)
    return [filename for filename in filenames if filename.endswith(extension)]


def mkdir_if_not_exists(default_save_path: str):
    """
    Make directory if not exists by a folder path
    :param default_save_path: str() -> path to create the folder
    :return: None
    """

    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)

def verify_existence_data(path: str):
    """
    Verify existence of a file by the folder path.
    :param path: location of folder
    :return: bool() -> existence of folder
    """

    if os.path.isfile(path):
        return True
    return False

def replace_multiple_inputs_str(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text