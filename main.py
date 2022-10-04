import torch as T
import argparse
import datetime
from src.trainer import PecanTrainer
from src.evaluator import PecanEvaluator
from src.traditional_ml_models import TraditionalML
from src.utils.functions import replace_multiple_inputs_str
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')

    #Project Parameterss

    parser.add_argument('--model', type=str,  default='SARIMAX',

                            help='Model of experiment, options: [LSTM, Linear, GRU, RNN, ConvRNN, FCN, TCN, ResNet, Transformer, MLP, TST, XGBoost, SVR, SARIMAX]')

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--debug_percent', type=float, default=0.2378)
    parser.add_argument('--task', type=str, default='traditional_models', help='Task of experiment, options: [train, test, traditional_models]')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length to the sequence training.')
    parser.add_argument('--output_length', required=False, type=int, default=1) #TODO {Implementar multiplas sequências}
    parser.add_argument('--seed', type=int, default=0, help='Seed used for deterministic results')

    #dataset parameters
    parser.add_argument('--root_path', type=str, default='data/', help='root path of the data file')
    parser.add_argument('--dataset', type=str, default='Pecanstreet', help='[Pecanstreet, HUE]')
    parser.add_argument('--resolution', type=str, default='1min', help='[1min, 1hour]')
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
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers to the Pytorch DataLoader.')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='Pin Memory to the Pytorch DataLoader.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for Pytorch DataLoader.')
    parser.add_argument('--n_epochs', type=int, default=201,
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
    parser.add_argument('--d_k', type=any, default=32,
                        help='Transformer number of heads.')
    parser.add_argument('--d_v', type=any, default=32,
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

    if args.task == 'train':
        trainer = PecanTrainer(args)
        trainer.train()
        print(f"[!] - {args.model} training completed")
    elif args.task == 'test':
        evaluator = PecanEvaluator(args)
        evaluator.evaluate()
        print(f"[!] - {args.model} evaluation completed")
    elif args.task == 'traditional_models':
        TraditionalML(args=args).SVRTest()
        TraditionalML(args=args).XGBoostTest()
        TraditionalML(args=args).statisticalModel()

        print(f"[!] - Evaluation of traditional models completed")
    else:
        raise NotImplemented("[?] - Task not implemented. Try using train, test or predict")

    print(f"[!] - Bye")

if __name__ == '__main__':
    main()

