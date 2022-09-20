import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from pandas import DataFrame
from cProfile import label
from typing import Tuple, List


def saveModelLosses(model_name: str, losses: Tuple[str, str, list], title: str, path: str, filename: str):
    plt.title(f"[`{model_name}`] - {title}")
    for type, marker, loss in losses:
        plt.plot(loss, marker, label=type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
  
    return f"[*] - Losses of {model_name} saved on {path}/{filename}"

def saveModelPreds(model_name: str, data: Tuple[list, str, str], title: str, path: str, filename:str):
    plt.title(f"[`{model_name}`] - {title}")
    for value, marker, label in data:
        plt.plot(value, marker, label=label)
    plt.xlabel(r'Time [h]')
    plt.ylabel(r'Energy [kW]')
    plt.legend()
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    return f"[*] - Preds of {model_name} saved on {path}/{filename}"


def saveModelMetrics(categories: List[str], data: Tuple[list, str, str, str], title: str, path: str, filename: str):
    label_lc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    plt.subplot(polar=True)
    print(data)
    for values, label, marker, color in data:
        plt.plot(label_lc, values, marker, c=color, label=label)
    lines, labels = plt.thetagrids(np.degrees(label_lc), labels=categories)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    return f"[*] - Metrics saved on {path}/{filename}"

def savePCACutOffThreshold(values, path, filename, title):
    plt.ylim(0.0,1.1)
    plt.plot(values, marker='o', linestyle='--', color='b')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red')
    plt.title(f"{title}")
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.grid(axis='x')
    return f"[*] - PCA cut-off thresholst saved on {path}/{filename}"

def savePCAHeatMap(df: DataFrame, path, filename):
    plt.figure(figsize=(20, 20))
    plt.title(f'[`PCA`] - Features components', fontsize=15)

    ax = sns.heatmap(df,  cmap="YlGnBu")
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('[PC] - Components', fontsize=15)
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    return f"[*] - PCA uniform heatmap created on {path}/{filename}"

def saveSHAPForce(explainer, shap_values, features_names, path, filename):
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], features_names, show=False, matplotlib=True)
    plt.style.use('fast')

    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    return f"[*] - SHAP force plot created on {path}/{filename}"

def saveSHAPSummaryPlot(shap_values, features, features_names, title, path, filename):
    shap.initjs()
    shap.summary_plot(shap_values[0, :, :], features=features, 
                  feature_names=features_names, 
                  plot_type='bar', show=False)
    plt.ylabel("Feature name")
    plt.xlabel("Average Impact")
    plt.title(title)
    plt.savefig(f'{path}/{filename}.svg', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{path}/{filename}.png', dpi=600, bbox_inches='tight')
    return f"[*] - SHAP summary plot created on {path}/{filename}"


