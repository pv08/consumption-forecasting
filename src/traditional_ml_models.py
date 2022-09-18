from dataclasses import dataclass
import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy import stats
from src.utils.functions import mkdir_if_not_exists, save_json_metrics, descale
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.utils.graphs import saveModelLosses, saveModelPreds
from src.pecan_wrapper.basic_wrapper import PecanWrapper

class TraditionalML(PecanWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.X_train, self.y_train = self.dataset.train_df.copy().drop('consumption', axis=1), self.dataset.train_df['consumption']
        self.X_validation, self.y_validation = self.dataset.val_df.copy().drop('consumption', axis=1), self.dataset.val_df['consumption']
        self.X_test, self.y_test = self.dataset.test_df.copy().drop('consumption', axis=1), self.dataset.test_df['consumption']

        self.descaler = MinMaxScaler(feature_range=(-1,1))
        
        scale_label_idx = list(self.dataset.scaler.feature_names_in_).index('consumption')
        self.descaler.min_ = self.dataset.scaler.min_[scale_label_idx]
        self.descaler.scale_ = self.dataset.scaler.scale_[scale_label_idx]


    def SVRTest(self):
        C_arr = [0.1,10,100,1000]
        eps_arr = [1,0.1,0.00001]

        # hyper_arr = []
        # hyper_cols =  ['c','epsilon','correlation']
        # for C in C_arr:
        #     for epsilon in eps_arr:
        #         model = SVR(kernel='rbf', C=C, epsilon=epsilon)
        #         svr = model.fit(self.X_train, self.y_train)
        #         y_pred = model.predict(self.X_test)
        #         corr = stats.pearsonr(self.y_test,y_pred)[0]
        #         hyper_arr.append([C,epsilon,corr])

        # hyper_df = pd.DataFrame(hyper_arr,columns= hyper_cols)
        # hyper_df = hyper_df.sort_values(by=['correlation'], ascending=False)



        model_svr = SVR(kernel='rbf', C=1000, epsilon=0.1, verbose=True)
        svr = model_svr.fit(self.X_train, self.y_train)
        y_preds = model_svr.predict(self.X_test) 

        result = [{
                    'test|MAE': mean_absolute_error(self.y_test, y_preds),
                    'test|MAPE': mean_absolute_percentage_error(self.y_test, y_preds),
                    'test|MSE': mean_squared_error(self.y_test, y_preds),
                    'model': 'SVR'
                }]
        save_json_metrics(content=result, path=self.local_result_dir, filename='metrics_report', model='SVR')

        test_preds = []
        for preds, labels in zip(list(y_preds), self.y_test.to_list()):
            test_preds.append(dict(
                model='SVR',
                label=float(labels),
                model_output=float(preds)
            ))
        model_infer = pd.DataFrame(test_preds)
        model_infer.label = descale(self.descaler, model_infer.label)
        model_infer.model_output = descale(self.descaler, model_infer.model_output)
        model_infer.to_csv(f"{self.local_result_dir}/SVR.csv")
        preds = [(model_infer.label[-72:].values, '-.', 'Real'), (model_infer.model_output[-72:].values, '-', 'Prediction')]
        saveModelPreds(model_name="SVR", 
                        data=preds, 
                        title="Descaled consumption predictions", 
                        path=self.local_imgs_dir, 
                        filename='last_72_h-predictions')




    def XGBoostTest(self):
        xg_regressor = XGBRegressor(seed=self.args.seed, objective='reg:squarederror',
                            gamma=1, 
                            learning_rate=0.1, 
                            max_depth=4, 
                            reg_lambda=0, 
                            scale_pos_weight=1)
        xg_regressor.fit(self.X_train, self.y_train, 
                 eval_set=[(self.X_train, self.y_train), (self.X_validation, self.y_validation)], 
                 eval_metric=['rmse', 'mae', 'mape'],
                
                 verbose=True)
        y_preds = xg_regressor.predict(self.X_test)
        result = [{
            'test|MAE': mean_absolute_error(self.y_test, y_preds),
            'test|MAPE': mean_absolute_percentage_error(self.y_test, y_preds),
            'test|MSE': mean_squared_error(self.y_test, y_preds),
            'model': 'XGBoost'
        }]

        save_json_metrics(content=result, path=self.local_result_dir, filename='metrics_report', model='XGBoost')
        descaler = MinMaxScaler(feature_range=(-1,1))
        
        scale_label_idx = list(self.dataset.scaler.feature_names_in_).index('consumption')
        descaler.min_ = self.dataset.scaler.min_[scale_label_idx]
        descaler.scale_ = self.dataset.scaler.scale_[scale_label_idx]

        test_preds = []
        for preds, labels in zip(list(y_preds), self.y_test.to_list()):
            test_preds.append(dict(
                model='XGBoost',
                label=float(labels),
                model_output=float(preds)
            ))
        model_infer = pd.DataFrame(test_preds)
        model_infer.label = descale(descaler, model_infer.label)
        model_infer.model_output = descale(descaler, model_infer.model_output)
        model_infer.to_csv(f"{self.local_result_dir}/XGBoost.csv")
        preds = [(model_infer.label[-72:].values, '-.', 'Real'), (model_infer.model_output[-72:].values, '-', 'Prediction')]
        saveModelPreds(model_name="XGBoost", 
                        data=preds, 
                        title="Descaled consumption predictions", 
                        path=f"{self.local_imgs_dir}/{self.args.model}", 
                        filename='last_72_h-predictions')


        evals_results = xg_regressor.evals_result()
        train_evals_results = dict(evals_results['validation_0'])
        train_evals_results['mse'] = [rmse_error ** 2 for rmse_error in train_evals_results['rmse']]
        validation_evals_results = dict(evals_results['validation_1'])
        validation_evals_results['mse'] = [rmse_error ** 2 for rmse_error in validation_evals_results['rmse']]

        losses = [('Train', '-.' ,train_evals_results['mse']), ('Validation', '-', validation_evals_results['mse'])]
        saveModelLosses(model_name='XGBoost', losses=losses, title='MSE Losses', path=f"{self.local_imgs_dir}/{self.args.model}", filename='train_val_losses')

        bst = xg_regressor.get_booster()
        for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
            print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

        node_params = {'share': 'box',
                    'style': 'filled, rounded',
                    'fillcolor': '#78cbe'}

        leaf_params = {'shape': 'box',
                    'style': 'filled',
                    'fillcolor': '#e48038'}

        render = xg.to_graphviz(xg_regressor, num_trees=0, size="10,10",
                    condition_node_params=node_params,
                    leaf_node_params=leaf_params)
        render.graph_attr = {'dpi':'600'}
        render.render(f'{self.local_imgs_dir}/{self.args.model}/xgboost_tree', format = 'svg')
        render.render(f'{self.local_imgs_dir}/{self.args.model}/xgboost_tree', format = 'eps')
        render.render(f'{self.local_imgs_dir}/{self.args.model}/xgboost_tree', format = 'png')

