import numpy as np


class MicroReactorSimulator:
    """
    A really dumb microreactor simulator.
    """

    _Rho_d0 = {
        8: -0.033085599,
        4: -0.033085599,
        2: -0.033085599 + 0.0071,
        1: -0.033085599 + 0.0071 + 0.0082
    }

    _Reactivity_per_degree = {
        8: 26.11e-5,
        4: 16.11e-5,
        2: 7.33e-5,
        1: 2.77e-5
    }

    _critical_setpoint = {
        8: 77.56,
        4: 125.56,
        2: 177.84,
        1: 179.0
    }

    _betas = np.array([1.42481e-4,
                       9.24281e-4,
                       7.79956e-4,
                       2.06583e-3,
                       6.71175e-4,
                       2.17806e-4])

    _lambdas = np.array([1.272e-2,
                         3.174e-2,
                         1.160e-1,
                         3.110e-1,
                         1.400,
                         3.870])


    Lambda = 1.68e-3 # neutron lifetime
    beta = 0.0048

    def __init__(self, num_drums=8, d_time=0.01):
        self.num_drums = num_drums
        self.d_time = d_time
        self.time = 0
        self.Rho_d0 = self._Rho_d0[num_drums]
        self.Reactivity_per_degree = self._Reactivity_per_degree[num_drums]
        self.drum_position = self._critical_setpoint[num_drums]
        self.power = 100
        # somewhat arbitrary initial precursor concentrations chosen based on running things for a while
        self.precursors = np.array([1.2, 3.0, 0.7, 0.7, 0.05, 0.01])
        self.precursors = self._betas*100 / (self._lambdas)
        self.reactivity_inserted = 0
        self.drum_history = [self.drum_position]
        self.action_history = [0]
        self.power_history = [100]
        self.time_history = [0]


    def iterate(self):
        dPower = self.power * (self.reactivity_inserted - self.beta) / self.Lambda
        dPower += np.sum(self._lambdas * self.precursors) / self.Lambda
        dPrecursors = self.power * self._betas - self._lambdas * self.precursors
        self.power += dPower * self.d_time
        self.precursors += dPrecursors * self.d_time

    def step(self, action, step_size=1):
        self.reactivity_inserted += action * self.Reactivity_per_degree
        self.drum_position += action
        for _ in range(int(1 / self.d_time)*step_size):
            self.iterate()
        
        self.drum_history.append(self.drum_position)
        self.action_history.append(action)
        self.power_history.append(self.power.item())
        self.time += step_size
        self.time_history.append(self.time)
        return self.power, self.precursors

    def zero_reactivity(self):
        self.reactivity_inserted = 0