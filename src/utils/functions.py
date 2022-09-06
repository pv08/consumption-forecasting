import torch as T
import os
import numpy as np
import wwo_hist
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
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
    plt.savefig(f'etc/imgs/predictions/{participant}_{model}_model.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'etc/imgs/predictions/{participant}_{model}_model.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'etc/imgs/predictions/{participant}_{model}_model.png', dpi=600, bbox_inches='tight')




def _regressor_ensemble_class_dict(args):
    args.ensemble_dict_ckpt = dict(zip(args.ensemble_models, _get_multiples_best_epochs(args)))

    ensemble_models = []
    for model in args.ensemble_models:
        tmp_args = args
        tmp_args.model = model
        ensemble_models.append(_regressor_eval_class_dict(tmp_args, args.ensemble_dict_ckpt[model], args.scaler))
    model = RecorrentEnsembleRegressor(device=args.device, lr=args.lr, ModelArray=ensemble_models)
    return model



def _get_resume_and_best_epoch(task, participant_id, activation_fn, model):
    resume_ckpt = None
    number_last_epoch = None

    if task == 'train':
        try:
            list_epochs = next(os.walk(
                f'lib/ckpts/participants/{participant_id}/{activation_fn}/{model}/epochs/'))[2]
        finally:
            if len(list_epochs) > 0:
                last_epoch = list_epochs[len(list_epochs) - 1]
                number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
                print(f"[!] - Last Epoch loaded - {number_last_epoch}")
                resume_ckpt = f'lib/ckpts/participants/{participant_id}/{activation_fn}/{model}/epochs/{last_epoch}'

    elif task in ['test', 'predict', 'ensemble']:
        try:
            list_best_epochs = next(os.walk(
                f'lib/ckpts/participants/{participant_id}/{activation_fn}/{model}/best/'))[2]
        finally:
            if len(list_best_epochs) > 0:
                last_epoch = list_best_epochs[len(list_best_epochs) - 1]
                number_last_epoch = last_epoch[last_epoch.find("=") + 1: last_epoch.find("=") + 4]
                print(f"[!] - Best Epoch loaded - {number_last_epoch}")
                resume_ckpt = f'lib/ckpts/participants/{participant_id}/{activation_fn}/{model}/best/{last_epoch}'
    else:
        raise NotImplementedError(f"[?] - Task not implemented")
    return resume_ckpt, number_last_epoch

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

def write_json_file(value, filepath, filename):
    with open(f'{filepath}/{filename}.json', 'w') as json_file:
        json.dump(value, json_file)

def read_json_file(filepath, filename):
    with open(f'{filepath}/{filename}.json', 'r') as json_file:
        data = json.load(json_file)
        json_file.close()
    return data


def write_test_json(path, result, model, task):
    mkdir_if_not_exists(f'{path}/{task}')

    try:
        with open(f'{path}/{task}/result_report.json') as json_file:
            data = json.load(json_file)
            json_file.close()

        for model_dict in data:
            if model_dict['model'] == model:
                model_dict[task] = result
                write_json_file(data, f'{path}/{task}', 'result_report')
                return None
        data.append({'model': model, task: result})
        write_json_file(data, f'{path}/{task}', 'result_report')
        print(f"[!] - {task.capitalize()} report of {model} model saved")
        return None
    except:
        result_value = {
            'model': model,
            task: result
        }
        write_json_file([result_value], f'{path}/{task}', 'result_report')
        print(f"[!] - {task.capitalize()} report of {model} model saved")


def MSEError(labels, predictions):
    return np.sum(np.diff([labels, predictions], axis=0) ** 2) / len(predictions)


def MAEError(labels, predictions):
    return np.sum(abs(np.diff([labels, predictions], axis=0))) / len(predictions)


