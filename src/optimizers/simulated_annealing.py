import numpy as np
from pyexpat import model
from math import exp
from typing import Dict, List
from random import uniform
from sklearn.metrics import mean_squared_error
from src.utils.functions import write_json_file, mkdir_if_not_exists

class SA:
    def __init__(self, initial_weights: Dict[str, float], 
                model_predictions: Dict[str, List], 
                default_label: List[float],
                filepath: str, 
                t_0: float=1000, 
                t_f: float=1e-20,
                patience=1000, minTempI: int=200, energyDiff: float=0.01, tempAlpha: float=0.95):
        self.t = t_0
        self.t_f = t_f
        self.filepath = filepath
        self.default_label = default_label
        self.current_weights = dict(sorted(initial_weights.items()))
        self.model_predictions = dict(sorted(model_predictions.items()))
        self.patience = patience
        self.minTempI = minTempI
        self.energyDiff = energyDiff
        self.tempAlpha = tempAlpha
        
        self.current_energy = self.verifySysEnergy(model_predictions=self.model_predictions, weights=self.current_weights, label=self.default_label)
        self.best_energy = [self.current_energy]
        self.hist = []
        self.best_energy_hist = []
        

    def opt(self):
        tries = 0
        k = 0
        while (self.t > self.t_f) and (tries < self.patience):
            
            thermalBalance_t = False
            i = 0
            tmpEnergy = []
            while (not thermalBalance_t) and (tries < self.patience):
                new_neighbor = self.generateNewWeights(self.current_weights)
                E_n = self.verifySysEnergy(model_predictions=self.model_predictions, weights=new_neighbor, label=self.default_label)
                dE = self.current_energy - E_n
                if dE > 0:
                    self.current_weights = new_neighbor
                    tmpEnergy.append(E_n)
                    self.current_energy = E_n
                    self.hist.append( {'temp_i':i, 'temp':self.t, 'energy':E_n} )
                    # print(f"[I-{i}] - Temp: {self.t} | Energy: {E_n}")
                    i += 1
                    k += 1
                else:
                    p = self.safeExp(dE / self.t )
                    randVal = uniform(-1, 1)
                    if (randVal > 0) and (randVal < p):
                        self.current_weights = new_neighbor
                        tmpEnergy.append(E_n)
                        self.current_energy = E_n
                        self.hist.append( {'temp_i':i, 'temp':self.t, 'energy':E_n} )
                        # print(f"[I-{i}] - Temp: {self.t} | Energy: {E_n}")
                        i += 1
                        k += 1
                if self.current_energy < min(self.best_energy):
                    self.best_energy.append(self.current_energy)
                    self.best_energy_hist.append({'global_iteration': k, 'energy': self.current_energy, 'temp': self.t})
                if (i % self.minTempI == 0) and (i > 0):
                    avgTmpEnergy = np.average(tmpEnergy[-i: -1]) # ignorando o adicionado
                    if (avgTmpEnergy - self.current_energy) < self.energyDiff:
                        thermalBalance_t = True
                        print(f"[!] - Thermal balance reached for temp {self.t}. | Best Energy: {min(self.best_energy)} | Try: {tries}-{self.patience}")
                        self.t *= self.tempAlpha
                        # print(f"[!] - Temp decreased to {self.t}")
                    else:
                        # print(f"[?] - Thermal Balance not reached. Try: {tries}-{self.patience}")
                        tries += 1
        
        write_json_file(value=self.current_weights, filepath=self.filepath, filename='validation_weights')
        write_json_file(value=self.hist, filepath=self.filepath, filename='global_hist')
        write_json_file(value=self.best_energy_hist, filepath=self.filepath, filename='best_hist')
        print(f"******** Thermal Balance reached at {self.t} - {min(self.best_energy)}")
                    

    @staticmethod
    def safeExp(x: float):
        try: 
            return exp(x)
        except:
            return 0
    
    
    def verifySysEnergy(self, model_predictions: Dict[str, List], weights: Dict[str, float], label: List[float]) -> float:
        avg_preds = np.average([*model_predictions.values()], axis=0, weights=[*weights.values()])
        return mean_squared_error(label, avg_preds)

    
    def generateNewWeights(self, weights: Dict[str, float]) -> Dict[str, float]:
        listWeights = [*weights.values()]
        while True:
            p1, p2 = np.random.randint(0, len(weights)), np.random.randint(0, len(weights))
            if p1 != p2: break
        v = np.random.uniform(0, listWeights[p2])

        listWeights[p1] = min(1, listWeights[p1] + v)
        listWeights[p2] = max(0, listWeights[p2] - v)
        weights = dict(zip(weights.keys(), listWeights))
        return weights
