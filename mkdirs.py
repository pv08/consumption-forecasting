from argparse import ArgumentParser
from pyexpat import model
from src.utils.functions import mkdir_if_not_exists
def main():
    models = ['LSTM', 'GRU', 'RNN', 'ConvRNN', 'FCN', 'TCN', 'ResNet', 'Transformer', 'MLP', 'TST']
    resolutions = ['1min', '15min', '1hour']
    datasets = [('Pecanstreet', '661'), ('HUE', '1')]
    types = ['all', 'PCA', 'SHAP']
    parser = ArgumentParser(description='[Pecan Street Dataport] Forecasting the energy consumption of Pecan Street')

    for type in types:
        for dataset, id in datasets:
            for resolution in resolutions:
                for model in models:
                    task = 'single-step'
                    #Create etc folder for directory componentes
                    mkdir_if_not_exists('etc/')
                    # Create checkpoints folders for training
                    mkdir_if_not_exists('etc/ckpts/')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}/{resolution}')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}/{model}')
                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}/{model}/epochs')
                    every_ckpt_location = f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}/{model}/epochs'
                    every_ckpt_filename = f'{task}-{model}-ckpt-{dataset}-participant-id-{id}' + "_{epoch:03d}"

                    mkdir_if_not_exists(f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}/{model}/best')
                    best_ckpt_location = f'etc/ckpts/{dataset}/{task}/{id}/{resolution}/{type}/{model}/best'
                    best_ckpt_filename = f'best-{task}-{model}-ckpt-{dataset}-participant-id-{id}' + "_{epoch:03d}"

                    #Create log folder for training
                    mkdir_if_not_exists('etc/log/')
                    mkdir_if_not_exists(f'etc/log/{dataset}/')
                    mkdir_if_not_exists(f'etc/log/{dataset}/{task}')
                    mkdir_if_not_exists(f'etc/log/{dataset}/{task}/{id}')
                    mkdir_if_not_exists(f'etc/log/{dataset}/{task}/{id}/{resolution}')
                    mkdir_if_not_exists(f'etc/log/{dataset}/{task}/{id}/{resolution}/{type}')
                    mkdir_if_not_exists(f'etc/log/{dataset}/{task}/{id}/{resolution}/{type}/{model}')
                    local_logger_dir = f'etc/log/{dataset}/{task}//{id}/{resolution}/{type}/{model}'

                    #Create results folders
                    mkdir_if_not_exists('etc/results/')
                    mkdir_if_not_exists(f'etc/results/{dataset}')
                    mkdir_if_not_exists(f'etc/results/{dataset}/{task}')
                    mkdir_if_not_exists(f'etc/results/{dataset}/{task}/{id}')
                    mkdir_if_not_exists(f'etc/results/{dataset}/{task}/{id}/{resolution}')
                    mkdir_if_not_exists(f'etc/results/{dataset}/{task}/{id}/{resolution}/{type}')
                    mkdir_if_not_exists(f'etc/results/{dataset}/{task}/{id}/{resolution}/{type}/{model}')
                    local_result_dir = f'etc/results/{dataset}/{task}/{id}/{resolution}/{type}'


                    #Create img folders for validation
                    mkdir_if_not_exists('etc/imgs/')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}/{task}')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}/{task}/{id}')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}/{task}/{id}/{resolution}')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}/{task}/{id}/{resolution}/{type}')
                    mkdir_if_not_exists(f'etc/imgs/{dataset}/{task}/{id}/{resolution}/{type}/{model}')
                    local_imgs_dir = f'etc/imgs/{dataset}/{task}/{id}/{resolution}/{type}'
                    print(type, dataset, id, resolution, model)

if __name__ == "__main__":
    main()