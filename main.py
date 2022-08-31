import torch as T
import argparse
import datetime
from src.trainer import PecanTrainer
from src.evaluator import PecanEvaluator
from src.validator import PecanValidator
from src.ensemble import PecanEnsemble
from src.utils.functions import replace_multiple_inputs_str
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')

    parser.add_argument('--model', type=str,  default='FCN',
                            help='Model of experiment, options: [LSTM, Linear, GRU, RNN, ConvRNN, FCN, TCN, ResNet, Transformer, MLP, TST, RecorrentEnsemble]')

    parser.add_argument('--ensemble', type=bool,  default=False)
    parser.add_argument('--ensemble_method', type=str, default='DeepEnsemble',
                        help="options: [Fusion, Voting, Bagging, GradientBoosting, NTE, "
                             "SE, AT, FGE, SGB, DeepEnsemble]")

    parser.add_argument('--ensemble_models', type=list,  default=['LSTM', 'GRU'])


    parser.add_argument('--task', type=str, default='train',
                        help='Task of experiment, options: [train, predict, test, ensemble, validate]')


    parser.add_argument('--participant_id', type=str, default='661_test_30_all', help='Pecan Street participant id')
    parser.add_argument('--root_path', type=str, default='data/participants_data/1min', help='root path of the data file')

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed used for deterministic results')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='Flag to use GPU or not.')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Flag to use GPU or not.')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Sequence length to the sequence training.')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='Hidden layers to the LSTM model.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Layers to the LSTM model.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Layers to the LSTM model.')

    parser.add_argument('--activation_fn', type=str, default='sigmoid',
                        help='Activation Functions. [sigmoid, gelu, relu]')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--num_workers', type=int, default=0,
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

    parser.add_argument('--tst_activation_fn', type=str, default='gelu',
                        help='Activation Functions. [gelu, relu]')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer lattent dim.')
    parser.add_argument('--n_head', type=int, default=16,
                        help='Transformer number of heads.')
    parser.add_argument('--d_ffn', type=int, default=256,
                        help='Transformer number of heads.')
    parser.add_argument('--max_seq_len', type=int, default=120,
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


    T.cuda.empty_cache()

    args.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    if args.task == 'train':
        trainer = PecanTrainer(args)
        trainer.train()
    elif args.task == 'test':
        evaluator = PecanEvaluator(args)
        evaluator.evaluate()
    elif args.task == 'validate':
        evaluator = PecanValidator(args)
        evaluator.validator()
    elif args.task == 'ensemble':
        ensemble = PecanEnsemble(args)
        ensemble.ensemble()

    else:
        raise NotImplemented("[?] - Task not implemented. Try using train, test or predict")


if __name__ == '__main__':
    main()

