from fileinput import filename
from io import FileIO
from os import mkdir
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from argparse import ArgumentParser
from src.utils.functions import mkdir_if_not_exists
from src.utils.graphs import savePCACutOffThreshold, savePCAHeatMap

def main():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--task', type=str, default='single-step', help=['single-step', 'multi-step'])
    parser.add_argument('--dataset', type=str, default='HUE', help=['Pecanstreet', 'HUE'])
    parser.add_argument('--resolution', type=str, default='1hour', help=['1min', '15min', '1hour'])
    parser.add_argument('--id', type=str, default='1')
    args = parser.parse_args()

    filename_dict = {
        'Pecanstreet': f'{args.id}_all_features.csv',
        'HUE': f'residential_{args.id}.csv'
    }
    pca_filename_dict = {
        'Pecanstreet': f'{args.id}_pca_features.csv',
        'HUE': f'residential_{args.id}.csv'
    }

    try:
        df = pd.read_csv(f"{args.root}/{args.dataset}/participants_data/{args.resolution}/features/all/{filename_dict[args.dataset]}")
    except:
        raise FileExistsError ("[?] - Make sure that preprocessed dataset exists")
    df.fillna(0, inplace=True)
    original_data = df.copy()

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(df)
    scaled_df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    features_names = [x for x in scaled_df.columns if x != "consumption"]
    features_values = scaled_df[features_names].values
    target_values = np.array(scaled_df['consumption'])
    pca = PCA(n_components = 0.99)
    features_pca = pca.fit(scaled_df)
    loadings = features_pca.components_
    mkdir_if_not_exists(f'etc/imgs/{args.dataset}/{args.task}/{args.id}/{args.resolution}/PCA-Study')
    local_pca_path = f'etc/imgs/{args.dataset}/{args.task}/{args.id}/{args.resolution}/PCA-Study'
    # savePCACutOffThreshold(np.cumsum(pca.explained_variance_ratio_), 
    #                                 path=local_pca_path, 
    #                                 filename=f'{args.dataset}-{args.resolution}_pca_cut_off_threshold',
    #                                 title='Best number of PCA components')
    pca_result_df = pd.DataFrame(pca.components_, columns = df.columns)

    # savePCAHeatMap(pca_result_df, local_pca_path, f"{args.dataset}-{args.resolution}-PCA_heatmap")
    n_pcs= pca.n_components_ 
    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = df.columns
    # get the most important feature names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    most_important_names.append('consumption')
    most_important_name = list(dict.fromkeys(most_important_names))
    importante_features_df = original_data[most_important_names]
    try:
        importante_features_df.to_csv(f"{args.root}/{args.dataset}/participants_data/{args.resolution}/features/PCA/{pca_filename_dict[args.dataset]}")
        print("[!] - Feature importance extraction by PCA concluded. Saved on", 
                f"{args.root}/{args.dataset}/participants_data/{args.resolution}/features/PCA/{pca_filename_dict[args.dataset]}")
    except:
        raise FileNotFoundError
if __name__ == "__main__":
    main()