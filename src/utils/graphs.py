from cProfile import label
import matplotlib.pyplot as plt
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
