import torch as T
import argparse
import datetime
import pandas as pd
import numpy as np
import shap
from src.trainer import PecanTrainer
from src.evaluator import PecanEvaluator
from src.traditional_ml_models import TraditionalML
from src.utils.functions import replace_multiple_inputs_str, mkdir_if_not_exists
from src.utils.graphs import saveSHAPForce, saveSHAPSummaryPlot
from torch.utils.data import DataLoader
from src.dataset import PecanDataset


from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')

    #Project Parameterss
    parser.add_argument('--model', type=str,  default='LSTM',
                            help='Model of experiment, options: [LSTM, Linear, GRU, RNN, ConvRNN, FCN, TCN, ResNet, Transformer, MLP, TST, XGBoost, SVR]')

    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--debug_percent', type=float, default=0.237)
    parser.add_argument('--task', type=str, default='train', help='Task of experiment, options: [train, test, traditional_models]')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length to the sequence training.')
    parser.add_argument('--output_length', required=False, type=int, default=1) #TODO {Implementar multiplas sequências}
    parser.add_argument('--seed', type=int, default=0, help='Seed used for deterministic results')

    #dataset parameters
    parser.add_argument('--root_path', type=str, default='data/', help='root path of the data file')
    parser.add_argument('--dataset', type=str, default='Pecanstreet', help='[Pecanstreet, HUE]')
    parser.add_argument('--resolution', type=str, default='1hour', help='[1min, 1hour]')
    parser.add_argument('--participant_id', type=str, default='661', help='Pecan Street participant id')
    parser.add_argument('--data_type', type=str, default='all', help='[all, PCA, SHAP]]')

    #Recorrent neural networks hyperparameters
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='Flag to use GPU or not.')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='Hidden layers to the LSTM model.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Layers to the LSTM model.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Layers to the LSTM model.')

    #Hyperparameters
    parser.add_argument('--activation_fn', type=str, default='sigmoid',
                        help='Activation Functions. [sigmoid, gelu, relu]')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of workers to the Pytorch DataLoader.')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='Pin Memory to the Pytorch DataLoader.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for Pytorch DataLoader.')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--early_stopping', type=bool, default=False,
                        help='Able early stopping.')
    parser.add_argument('--patience', type=int, default=2,
                        help='If the early stopping is True, define the patience level. Default is 2.')

    #Transfos=rmers hyperparameters
    parser.add_argument('--tst_activation_fn', type=str, default='gelu',
                        help='Activation Functions. [gelu, relu]')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer lattent dim.')
    parser.add_argument('--n_head', type=int, default=16,
                        help='Transformer number of heads.')
    parser.add_argument('--d_ffn', type=int, default=256,
                        help='Transformer number of heads.')
    parser.add_argument('--max_seq_len', type=int, default=60,
                        help='Transformer number of heads.')
    parser.add_argument('--d_k', type=any, default=60,
                        help='Transformer number of heads.')
    parser.add_argument('--d_v', type=any, default=60,
                        help='Transformer number of heads.')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='Transformer number of heads.')
    parser.add_argument('--fc_dropout', type=float, default=0.1,
                        help='Transformer number of heads.')


    args = parser.parse_args()


    run_id = replace_multiple_inputs_str(str(datetime.datetime.now()), OrderedDict([(':', ''), ('-', ''), (' ', '')]))
    args.run_id = run_id[:len(run_id) - 7]

    args.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    args.gpu = T.cuda.device_count()

    evaluator = PecanEvaluator(args)

    shap_train_loader = DataLoader(
            PecanDataset(evaluator.dataset.train_sequences, args.device),
            
            batch_size=32,
            shuffle = False,
            num_workers=0)

    shap_test_loader = DataLoader(
                PecanDataset(evaluator.dataset.test_sequences, args.device),
                
                batch_size=10000,
                shuffle = False,
                num_workers=0)

    shap_train_batch = next(iter(shap_train_loader))
    shap_test_batch = next(iter(shap_test_loader))
    sequences = shap_train_batch["sequence"]
    labels = shap_train_batch["label"]
    background = sequences.to(args.device)

    model = evaluator.regressor.model

    explainer = shap.DeepExplainer(model, background)

    test_sequences = shap_test_batch["sequence"]
    test_labels = shap_test_batch["label"]
    shap_test = test_sequences.to(args.device)

    T.backends.cudnn.enabled = False
    shap_values = explainer.shap_values(shap_test)
    local_path = f"{evaluator.local_imgs_dir}/{args.model}/SHAP"
    mkdir_if_not_exists(local_path)
    saveSHAPForce(explainer=explainer, shap_values=shap_values, 
                    features_names=evaluator.dataset.original_data.columns.to_list(),
                    path=local_path,
                    filename=f"{args.model}_feature_impact")
    df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values[-1, :, :]), axis=0), 
    "stdev_abs_shap": np.std(np.abs(shap_values[-1, :, :]), axis=0), 
    "name": evaluator.dataset.original_data.columns.to_list()
    })
    important_values = df.sort_values("mean_abs_shap", ascending=False)[:11]
    important_columns = important_values.name.to_list()
    if 'consumption' not in important_columns:
        important_columns.append('consumption')
    orginal_shap_important = evaluator.dataset.original_data[important_columns]
    save_file = {
        'Pecanstreet': f'{args.participant_id}_shap_features_{args.model}.csv',
        'HUE': f'residential_{args.participant_id}_{args.model}.csv'
    }
    orginal_shap_important.to_csv(f'{args.root_path}/{args.dataset}/participants_data/{args.resolution}/features/SHAP/{save_file[args.dataset]}', index=False)
    print("[!] - SHAP important features saved on", f'{args.root_path}/{args.dataset}/participants_data/{args.resolution}/features/SHAP/{save_file[args.dataset]}')
    saveSHAPSummaryPlot(shap_values=shap_values, features=test_sequences[0, :,:], 
                        features_names=evaluator.dataset.original_data.columns.to_list(),
                        title=f"[`{args.model}`] - Feature Importance", path=local_path, filename=f"{args.model}_summary_plot")
    


    


if __name__ == "__main__":
    main()