import numpy as np
import pandas as pd
import json
import math
import statistics as st
from src.optimizers.simulated_annealing import SimulatedAnnealing
from src.utils.functions import MSEError
from random import randint, random


class RealSA:
    def __init__(self, guide_func, label, models_prediction, x0, model_names, cooling_schedule='linear',
                 step_max=0.1, t_min=0.00001, t_max=0.1, alpha=None):



        self.label = label
        self.guide_func = guide_func
        self.predictions = models_prediction
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.hist = []

        self.cost_func = self.EnsembleMSEFunc
        self.x0 = x0
        self.current_state = self.x0
        self.current_energy = self.EnsembleMSEFunc(guide_func, self.x0, label, models_prediction)
        self.best_state = self.current_state
        self.best_energy = self.current_energy

        self.step, self.accept = 1, 0

        global_entropy = [self.current_energy]
        while self.t >= self.t_min:
            self.entropy_thermal_balance = False
            entropy_k = []
            k = 0
            energy_k = 0
            while not self.entropy_thermal_balance:
                # passo 4 - Gerar visinhos
                proposed_neighbor = self.get_neighbor(self.current_state)
                E_n = self.cost_func(self.guide_func, proposed_neighbor, self.label, self.predictions)
                #passo 5 - Calcular delta entre a função objetivo do visinho gerado com a energia do estado anterior (temperatura anterior)
                dE = E_n - self.current_energy

                #passo 5a - Caso for positivo, o estado atual será admitido e a energia da temperatura será guardada
                if dE >= 0:
                    self.current_state = proposed_neighbor[:]
                    energy_k = E_n
                    entropy_k.append(E_n)
                    k += 1
                #passo 5b - Caso contrário gerar um valor aleatório e um valor de probabilidade p
                else:
                    p = self.safe_exp(-dE / self.t * 1.380649e-23)
                    random_value = random()
                    #passo 5bi - verificar se o valor aleatório está no intervalo [0,p]
                    if random_value >= 0 and random_value <= p:
                        self.current_state = proposed_neighbor[:]
                        energy_k = E_n
                        entropy_k.append(E_n)
                        k += 1
                #passo 6 - Verificar a 200 iterações de temperatura a média das energias admitidas para a temperatura e as globais aceitas
                if k % 200 == 0 and k > 0:
                    diff = st.mean(global_entropy) - st.mean(entropy_k)
                    if diff < 0.001:
                        self.current_energy = energy_k
                        global_entropy.append(self.current_energy)
                        #passo 6i - Verificar se o estado admitido é menor que o melhor já visto.
                        if self.current_energy < self.best_energy:
                            self.best_energy = self.current_energy
                            self.best_state = self.current_state
                        self.entropy_thermal_balance = True
                        self.t *= 0.95

            self.hist.append([self.step, self.t, self.current_energy, self.best_energy])
            print(
                f"Step: [{self.step}] | Temp: {self.t} | Energy: {self.best_energy}")
            self.step += 1

    def safe_exp(self, x):
        try:
            return math.exp(x)
        except:
            return 0

    @staticmethod
    def get_neighbor(current_state):
        neighbor = current_state.copy()

        while True:
            p1, p2 = np.random.randint(0, len(current_state)), np.random.randint(0, len(current_state))
            if p1 != p2: break

        v = np.random.uniform(0, current_state[p2])

        neighbor[p1] = min(1, current_state[p1] + v)
        neighbor[p2] = max(0, current_state[p2] - v)
        return neighbor

    @staticmethod
    def EnsembleMSEFunc(func, weights, labels, models_predictions):
        weight_sum = []

        for prediction, weight in zip(models_predictions, weights):
            weight_sum.append(prediction * weight)

        weight_avg = np.sum(weight_sum, axis=0) / sum(weights)

        return func(labels, weight_avg)

    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    def cooling(self):
        return self.t * 0.95



def main():
    with open(f'etc/results/validation/661_test_30_pca/result_report.json') as json_file:
        data = json.load(json_file)
        json_file.close()
    predictions_data = data
    new_prediction_data = []
    for model in predictions_data:
        for prediction in model['validate']:
            prediction['model'] = model['model']
        new_prediction_data += model['validate']
    complete_prediction_df = pd.DataFrame(new_prediction_data)
    labels = np.array(complete_prediction_df[complete_prediction_df['model'] == 'GRU'].label)

    models_predictions = []
    models = complete_prediction_df['model'].unique()
    for model in models:
        models_predictions.append(complete_prediction_df[complete_prediction_df['model'] == model].model_output)

    weights = [1/len(models_predictions) for i in range(len(models_predictions) - 1)]
    weights.append(1 - sum(weights))

    sa = RealSA(guide_func=MSEError, label=labels, models_prediction=np.array(models_predictions), x0=weights, model_names=models)
    print(sa)

if __name__ == "__main__":
    main()