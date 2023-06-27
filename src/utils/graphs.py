import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import statsmodels.api as sm
from pandas import DataFrame
from cProfile import label
from typing import Tuple, List, Any
from scipy import stats


def saveModelLosses(model_name: str, losses: Tuple[str, str, list], 
                    title: str, path: str, filename: str, show_title: bool = False, save_plots:bool=True, save_types:list = ['pdf']):
    if show_title: plt.title(f"[`{model_name}`] - {title}")
    for type, marker, loss in losses:
        plt.plot(loss, marker, label=type)
    plt.ylabel(r'Perda')
    plt.xlabel('Épocas')
    plt.legend()
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
  
    return f"[*] - Losses of {model_name} saved on {path}/{filename}"

def saveModelPreds(model_name: str, data: Tuple[list, str, str], title: str, path: str, filename:str, resolution:str, save_plots:bool=True, save_types:list = ['pdf']):
    plt.title(f"[`{model_name}`] - {title}")
    for value, marker, label in data:
        plt.plot(value, marker, label=label)
    plt.xlabel(f'Time [{resolution}]')
    plt.ylabel(r'Energy [kW]')
    plt.legend()
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')

    return f"[*] - Preds of {model_name} saved on {path}/{filename}"


def saveModelMetrics(categories: List[str], data: Tuple[list, str, str, str], title: str, path: str, filename: str, save_plots:bool=True, save_types:list = ['pdf']):
    categories = [*categories, categories[0]]
    label_lc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    plt.subplot(polar=True)
    print(data)
    for values, label, marker, color in data:
        values = [*values, values[0]]
        plt.plot(label_lc, values, marker, c=color, label=label)
    lines, labels = plt.thetagrids(np.degrees(label_lc), labels=categories)
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.15, 0.))
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')

    return f"[*] - Metrics saved on {path}/{filename}"

def savePCACutOffThreshold(data: List[Tuple[Any, str, str, str, str]], path, filename, title, save_plots:bool=True, save_types:list = ['pdf']):
    plt.ylim(0.0,1.1)
    for values, marker, style, color, legend in data:
        plt.plot(values, marker=marker, linestyle=style, color=color, label=legend)
    plt.xlabel('Número de componentes')
    plt.ylabel('Variância cumulativa');
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, 'Limite de confiança: 95%', color = 'red')
    plt.legend()
    plt.grid(axis='x')
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')

    
    return f"[*] - PCA cut-off thresholst saved on {path}/{filename}"

def savePCAHeatMap(df: DataFrame, path, filename, save_plots:bool=True, save_types:list = ['pdf']):
    plt.figure(figsize=(20, 20))
    plt.title(f'[`PCA`] - Features components', fontsize=15)

    ax = sns.heatmap(df,  cmap="YlGnBu")
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('[PC] - Components', fontsize=15)
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    return f"[*] - PCA uniform heatmap created on {path}/{filename}"

def saveSHAPForce(explainer, shap_values, features_names, path, filename, save_plots:bool=True, save_types:list = ['pdf']):
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], features_names, show=False, matplotlib=True)
    plt.style.use('fast')
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    plt.show()
    return f"[*] - SHAP force plot created on {path}/{filename}"

def saveSHAPSummaryPlot(shap_values, features, features_names, title, path, filename, save_plots:bool=True, save_types:list = ['pdf']):
    shap.initjs()
    shap.summary_plot(shap_values[0, :, :], features=features, 
                  feature_names=features_names, 
                  plot_type='bar', show=False)
    plt.ylabel("Atributo", fontdict={'size': 18})
    plt.xlabel("Impacto médio", fontdict={'size': 18})
    plt.xticks(fontsize=18)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=18)
    plt.title(title, fontdict={'size': 18})
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    plt.show()

    return f"[*] - SHAP summary plot created on {path}/{filename}"

def saveDatasetBoxplot(df: DataFrame, path: str, filename: str, save_plots:bool=True, save_types:list = ['pdf']):
    plt.figure(figsize=(20,10))
    df_box = df.boxplot(column=df.columns.to_list())
    df_box.plot()
    plt.ylabel('Valores normalizados')
    plt.xticks(rotation=45)
    plt.xlabel('Características')
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    return f"[*] - Box plot created on {path}/{filename}"

def saveQQPlot(data, path: str, filename: str, save_plots:bool=True, save_types:list = ['pdf']):
    sm.qqplot(data, fit=True, line='q')
    plt.xlabel('Quantil teórico')
    plt.ylabel('Quantil de amostra')
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    return f"[*] - QQ plot created on {path}/{filename}"

def saveMultiDistributionPDF(data: List[Tuple[Any, str]], path, filename, save_plots:bool=True, save_types:list = ['pdf']):
    for value, dataset in data:
        mean = value.mean()
        std = value.std()
        snd = stats.norm(mean, std)
        x = np.linspace(-2.0, 2.0, 1000)
        plt.plot(x, snd.pdf(x)/10, label=dataset)
    plt.legend()
    plt.xlabel('Valores aleatórios')
    plt.ylabel('PDF')
    if save_plots:
        for typs in save_types:
            plt.savefig(f'{path}/{filename}.{typs}', dpi=600, bbox_inches='tight')
    return f"[*] - PDF plot created on {path}/{filename}"





