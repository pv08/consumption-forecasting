import pandas as pd
from src.optimizers.simulated_annealing import SA
from src.utils.functions import mkdir_if_not_exists


def main():
    
    default_models = ['LSTM', 'GRU', 'RNN', 'ConvRNN', 'FCN', 'TCN', 'ResNet', 'Transformer', 'MLP', 'TST']
    default_models = dict(zip(default_models, [None for model in default_models]))
    default_label = None
    for res in ['1min', '15min', '1hour']:
        for typ in ['all', 'PCA', 'SHAP'] :
            for model in default_models:
                model_preds = pd.read_csv(f'etc/results/Pecanstreet/single-step/661/{res}/{typ}/{model}/validation_preds.csv')
                default_label = model_preds['label'].values
                default_models[model] = model_preds['model_output'].values
            init_weights = []
            init_weights = [1/len(default_models) for i in range(len(default_models) - 1)] # tirando o peso do último modelo
            init_weights.append(1 - sum(init_weights)) # peso do último será 1 - a soma do restante
            init_weights = dict(zip(default_models, init_weights))
            mkdir_if_not_exists(f'etc/log/Pecanstreet/single-step/661/{res}/{typ}/opt')
            print(f"[!] - Generating SA weights for Pecan street {model}/{res}/{typ}")
            sa = SA(initial_weights=init_weights, 
                        model_predictions=default_models, 
                        default_label=default_label, filepath=f'etc/log/Pecanstreet/single-step/661/{res}/{typ}/opt')

            sa.opt()
            print(f'[!] - Pecanstreet-{model}/{res}/{typ} | SA validation weights saved on etc/log/Pecanstreet/single-step/661/{res}/{typ}/opt')
    
    for typ in ['all', 'PCA', 'SHAP'] :
        for model in default_models:
            model_preds = pd.read_csv(f'etc/results/HUE/single-step/1/{res}/{typ}/{model}/validation_preds.csv')
            default_label = model_preds['label'].values
            default_models[model] = model_preds['model_output'].values
        init_weights = []
        init_weights = [1/len(default_models) for i in range(len(default_models) - 1)] # tirando o peso do último modelo
        init_weights.append(1 - sum(init_weights)) # peso do último será 1 - a soma do restante
        init_weights = dict(zip(default_models, init_weights))
        mkdir_if_not_exists(f'etc/log/HUE/single-step/1/{res}/{typ}/opt')
        print(f"[!] - Generating SA weights for HUE {model}/{res}/{typ}")
        sa = SA(initial_weights=init_weights, 
                    model_predictions=default_models, 
                    default_label=default_label, filepath=f'etc/log/HUE/single-step/1/{res}/{typ}/opt')

        sa.opt()
        print(f'[!] - HUE-{model}/{res}/{typ} | SA validation weights saved on etc/log/HUE/single-step/1/{res}/{typ}/opt')

if __name__ == '__main__':
    main()