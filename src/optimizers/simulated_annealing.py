from random import randint, random
import math
import numpy as np
class SimulatedAnnealing:
    def __init__(self, label, models_prediction, x0, opt_mode, cooling_schedule='linear',
                 step_max=1000, t_min=0, t_max=100, bounds=[], alpha=None, damping=1):
        assert opt_mode in ['combinatorial', 'continuous',
                            'base'], 'opt_mode must be either "combinatorial" or "continuous"'
        assert cooling_schedule in ['linear', 'exponential', 'logarithmic',
                                    'quadratic'], 'cooling_schedule must be either in ["linear", "exponential", "logarithmic", "quadratic"]'

        self.label = label
        self.predictions = models_prediction
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.opt_mode = opt_mode
        self.hist = []
        self.cooling_schedule = cooling_schedule

        self.cost_func = self.EnsembleMSEFunc
        self.x0 = x0
        self.bounds = bounds[:]
        self.damping = damping
        self.current_state = self.x0
        self.current_energy = self.EnsembleMSEFunc(self.x0, label, models_prediction)
        self.best_state = self.current_state
        self.best_energy = self.current_energy

        if self.opt_mode == 'combinatorial': self.get_neighbor = self.move_combinatorial
        if self.opt_mode == 'continuous': self.get_neighbor = self.move_continuous
        if self.opt_mode == 'base': self.get_neighbor = self.base_solution

        if self.cooling_schedule == 'linear':
            if alpha != None:
                self.update_t = self.cooling_linear_m
                self.cooling_schedule = 'linear multiplicative cooling'
                self.alpha = alpha
            if alpha == None:
                self.update_t = self.cooling_linear_a
                self.cooling_schedule = 'linear additive cooling'

        if self.cooling_schedule == 'quadratic':
            if alpha != None:
                self.update_t = self.cooling_quadratic_m
                self.cooling_schedule = 'quadratic multiplicative cooling'
                self.alpha = alpha
            if alpha == None:
                self.update_t = self.cooling_quadratic_a
                self.cooling_schedule = 'quadratic additive cooling'

        if self.cooling_schedule == 'exponential':
            if alpha == None:
                self.alpha = 0.8
            else:
                self.alpha = alpha
            self.update_t = self.cooling_exponential

        if self.cooling_schedule == 'logarithmic':
            if alpha == None:
                self.alpha = 0.8
            else:
                self.alpha = alpha
            self.update_t = self.cooling_logarithmic

        self.step, self.accept = 1, 0
        while self.step < self.step_max and self.t >= self.t_min and self.t > 0:
            proposed_neighbor = self.get_neighbor()

            E_n = self.cost_func(proposed_neighbor, self.label, self.predictions)
            dE = E_n - self.current_energy

            if random() < self.safe_exp(-dE / self.t):
                self.current_energy = E_n
                self.current_state = proposed_neighbor[:]
                self.accept += 1

            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor[:]
                self.best_ensemble_result = self.weight_avg

            self.hist.append([self.step, self.t, self.current_energy, self.best_energy])

            self.t = self.update_t(self.step)
            self.step += 1

        self.acceptance_rate = self.accept / self.step

    def base_solution(self):
        neighbor = self.current_state.copy()
        p1, p2 = np.random.randint(0, len(self.current_state)), np.random.randint(0, len(self.current_state))
        v = np.random.uniform(0, self.current_state[p2])

        neighbor[p1] = min(1, self.current_state[p1] + v)
        neighbor[p2] = max(0, self.current_state[p2] - v)
        return neighbor

    def move_continuous(self):
        neighbor = [item + ((np.random.uniform(0, 1) - 0.5) * self.damping) for item in self.current_state]

        if self.bounds:
            for i in range(len(neighbor)):
                x_min, x_max = self.bounds[i]
                neighbor[i] = min(max(neighbor[i], x_min), x_max)
        return neighbor

    def move_combinatorial(self):
        p0 = randint(0, len(self.current_state) - 1)
        p1 = randint(0, len(self.current_state) - 1)

        neighbor = self.current_state[:]
        neighbor[p0], neighbor[p1] = neighbor[p1], neighbor[p0]

        return neighbor

    def cooling_linear_m(self, step):
        return self.t_max / (1 + self.alpha * step)

    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step) / self.step_max)

    def cooling_quadratic_m(self, step):
        return self.t_min / (1 + self.alpha * step ** 2)

    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.t_max - step) / self.step_max) ** 2

    def cooling_exponential_m(self, step):
        return self.t_max * self.alpha ** step

    def cooling_logarithmic_m(self, step):
        return self.t_max / (self.alpha * math.log(step + 1))

    def safe_exp(self, x):
        try:
            return math.exp(x)
        except:
            return 0

    def EnsembleMSEFunc(self, weights, labels, models_predictions):
        weight_sum = []

        for prediction, weight in zip(models_predictions, weights):
            weight_sum.append(prediction * weight)

        self.weight_avg = np.sum(weight_sum, axis=0) / models_predictions.shape[0]

        return MSEError(labels, self.weight_avg)
