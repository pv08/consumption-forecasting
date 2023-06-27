from dataclasses import dataclass
from pyexpat import model
import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
        mkdir_if_not_exists(f"{self.local_result_dir}/SARIMAX")
        mkdir_if_not_exists(f"{self.local_result_dir}/SVR")
        mkdir_if_not_exists(f"{self.local_result_dir}/XGBoost")


    def saveDefaultMetrics(self, y_, y, model_name, task:str='val'):
        
        assert task in ['val', 'test'], "[!] - Make sure that you select the correct task. val or test"
        results = [{
            f'{task}|MAE': mean_absolute_error(y, y_),
            f'{task}|MAPE': mean_absolute_percentage_error(y, y_),
            f'{task}|MSE': mean_squared_error(y, y_),
            'model': model_name
        }]

        save_json_metrics(content=results, 
                            path=self.local_result_dir, 
                            filename=f'validation_metrics_report' if task == 'val' else 'metrics_report', 
                            model=model_name)
        model_preds = []
        for preds, labels in zip(list(y_), y.to_list()):
            model_preds.append(dict(
                model=model_name,
                label=float(labels),
                model_output=float(preds)
            ))
        infer = pd.DataFrame(model_preds)
        infer.label = descale(self.descaler, infer.label)
        infer.model_output = descale(self.descaler, infer.model_output)
        infer.to_csv(f"{self.local_result_dir}/{model_name}/validation_preds.csv" if task == 'val' else f"{self.local_result_dir}/{model_name}/test_preds.csv")

        if task == 'test':
            data = [(infer.label[-72:].values, '-.', 'Real'), (infer.model_output[-72:].values, '-', 'Prediction')]
            saveModelPreds(model_name=model_name, 
                        data=data, 
                        title="Descaled consumption predictions", 
                        path=self.local_imgs_dir, 
                        filename='last_72_h-predictions', resolution=self.args.resolution)

        print('[!] - Metrics generated successfully')
            



    
    
    def statisticalModel(self):
        model_train = SARIMAX(self.y_train,  order=(4,1,3), seasonal_order=(2,0,2,12)).fit()
        model_val = SARIMAX(self.y_validation,  order=(4,1,3), seasonal_order=(2,0,2,12)).fit(model_train.params)
        model_test = SARIMAX(self.y_test,  order=(4,1,3), seasonal_order=(2,0,2,12)).fit(model_train.params)
        validation_preds = model_val.predict(typ='levels')
        y_preds = model_test.predict(typ='levels')

        self.saveDefaultMetrics(validation_preds, self.y_validation, 'SARIMAX', task='val')
        self.saveDefaultMetrics(y_preds, self.y_test, 'SARIMAX', task='test')




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
        validation_preds = model_svr.predict(self.X_validation) 
        y_preds = model_svr.predict(self.X_test) 

        self.saveDefaultMetrics(validation_preds, self.y_validation, 'SVR', 'val')
        self.saveDefaultMetrics(y_preds, self.y_test, 'SVR', 'test')




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
        validation_preds = xg_regressor.predict(self.X_validation)
        y_preds = xg_regressor.predict(self.X_test)

        self.saveDefaultMetrics(validation_preds, self.y_validation, 'XGBoost', 'val')
        self.saveDefaultMetrics(y_preds, self.y_test, 'XGBoost', 'test')

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

