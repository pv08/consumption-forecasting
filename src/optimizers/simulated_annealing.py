import math
import numpy as np
from random import randint, random
class   SimulatedAnnealing:

    def __init__(self, guide_func, label, models_prediction, x0, model_names, cooling_schedule='linear',
                 step_max=10000, t_min=0, t_max=1000, alpha=None):

        assert cooling_schedule in ['linear', 'exponential', 'logarithmic', 'quadratic'], 'cooling_schedule must be either in ["linear", "exponential", "logarithmic", "quadratic"]'

        self.label = label
        self.guide_func = guide_func
        self.predictions = models_prediction
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.hist = []
        self.cooling_schedule = cooling_schedule

        if alpha != None:
            self.alpha = alpha
            self.cooling_dict = {
                'linear': self.cooling_linear_m,
                'quadratic': self.cooling_quadratic_m,
                'exponential': self.cooling_exponential_m,
                'logarithmic': self.cooling_logarithmic_m
            }
        else:
            self.cooling_dict = {
                'linear': self.cooling_linear_a,
                'quadratic': self.cooling_quadratic_a
            }
        self.cost_func = self.EnsembleMSEFunc
        self.x0 = x0
        self.current_state = self.x0
        self.current_energy = self.EnsembleMSEFunc(guide_func, self.x0, label, models_prediction)
        self.best_state = self.current_state
        self.best_energy = self.current_energy

        self.step, self.accept = 1, 0
        while self.step < self.step_max and self.t >= self.t_min and self.t > 0:
            proposed_neighbor = self.get_neighbor(self.current_state)

            E_n = self.cost_func(self.guide_func, proposed_neighbor, self.label, self.predictions)
            dE = E_n - self.current_energy

            if random() < self.safe_exp(-dE / self.t):
                self.current_energy = E_n
                self.current_state = proposed_neighbor[:]
                self.accept += 1

            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor[:]

            self.hist.append([self.step, self.t, self.current_energy, self.best_energy])

            self.t = self.cooling_dict[self.cooling_schedule](self.step)
            self.step += 1
            print(f"Step: [{self.step}/{self.step_max}] | Temp: {self.t} | Weights Sum: {round(sum(self.best_state), 2)}")

        self.acceptance_rate = self.accept / self.step


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

    def cooling_linear_m(self, step):
        try:
            return self.t_max / (1+self.alpha * step)
        except:
            raise ValueError("[!] - Make sure that you set a value to alpha")

    #ARRUMAR O COLLING

    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    def cooling_quadratic_m(self, step):
        try:
            return self.t_min / (1 + self.alpha * step**2)
        except:
            raise ValueError("[!] - Make sure that you set a value to alpha")

    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.t_max - step) / self.step_max)**2

    def cooling_exponential_m(self, step):
        try:
            return self.t_max * self.alpha**step
        except:
            raise ValueError("[!] - Make sure that you set a value to alpha")

    def cooling_logarithmic_m(self, step):
        try:
            return self.t_max / (self.alpha * math.log(step + 1))
        except:
            raise ValueError("[!] - Make sure that you set a value to alpha")


    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'guide function: {self.guide_func}')
        print(f'  initial temp: {self.t_max}')
        print(f'    final temp: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        print(f'  final energy: {self.best_energy:0.6f}\n')
        print(f'  best weights: {self.best_state}\n')
        print(f'   weights sum: {sum(self.best_state):0.6f}\n')
        print('+-------------------------- END ---------------------------+')
