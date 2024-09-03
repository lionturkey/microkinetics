import numpy as np
import gymnasium as gym


class PIDController:
    """
    A PID controller.
    Attributes:
        Kp: the proportional gain
        Ki: the integral gain
        Kd: the derivative gain
        Kaw: the anti windup gain
        T_C: the PID parameter
        max_rate: the maximum rate of change of the control signal
        mutiplier: the multiplier for the gains based on the number of drums
    """

    def __init__(self, Kp=1, Ki=1.5, Kd=0.001, Kaw=0.3, T_C=0.2, max_rate=0.5, multiplier=1.0):
        self.Kp = Kp * multiplier
        self.Ki = Ki * multiplier
        self.Kd = Kd * multiplier
        self.Kaw = Kaw * multiplier
        self.T_C = T_C
        self.max_rate = max_rate
        self.integral = 0.0
        self.err_prev = 0.0
        self.t_prev = 0.0
        self.deriv_prev = 0.0

    def update(self, t, measurement, setpoint):
        """
        Update the PID controller.
        Returns:
            the new control signal
        """
        err = setpoint - measurement
        del_t = t - self.t_prev
        self.t_prev = t
        self.integral += self.Ki * err * del_t
        self.deriv_prev = (err - self.err_prev + self.T_C * self.deriv_prev) / (del_t + self.T_C)
        self.err_prev = err
        command = self.Kp * err + self.integral + self.Kd * self.deriv_prev
        command_sat = np.clip(command, -self.max_rate, self.max_rate)
        return command_sat


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
        self.Rho_d0 = self._Rho_d0[num_drums]
        self.Reactivity_per_degree = self._Reactivity_per_degree[num_drums]
        self.drum_position = self._critical_setpoint[num_drums]
        self.power = 100
        # somewhat arbitrary initial precursor concentrations chosen based on running things for a while
        self.precursors = np.array([1.2, 3.0, 0.7, 0.7, 0.05, 0.01])
        self.reactivity_inserted = 0
        self.drum_history = [self.drum_position]
        self.action_history = [0]
        self.power_history = [100]


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
        self.power_history.append(self.power)
        return self.power, self.precursors

    def zero_reactivity(self):
        self.reactivity_inserted = 0



class MicroGym(gym.Env):
    # WORK IN PROGRESS
    def __init__(self, num_drums=8, d_time=0.1, episode_length=2000):
        self.num_drums = num_drums
        self.d_time = d_time
        self.episode_length = episode_length
        self.simulator = MicroReactorSimulator(num_drums, d_time)
        self.t = 0

    def step(self, action):
        if self.t >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        power, precursors = self.simulator.step(action)
        self.t += self.simulator.d_time


        if self.t >= self.episode_length:
            truncated = True
        else:
            truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass


# def main():
#     # create a microreactor simulator and a PID controller
#     sim = MicroReactorSimulator()
#     pid = PIDController()

#     # define the desired power profile
#     def desired_profile(t):
#         if t < 30:
#             return 1.0
#         elif t < 60:
#             return 0.8
#         elif t < 90:
#             return 1.0
#         elif t < 120:
#             return 0.8
#         elif t < 150:
#             return 1.0
#         else:
#             return 0.4

#     # loop to get the next reactor state and control action
#     times = []
#     powers = []
#     t = 0
#     while t < 180:
#         desired_power = desired_profile(t)
#         n_r, Cr1, Cr2, Cr3, Cr4, Cr5, Cr6, X, I, Tf, Tm, Tc = sim.state
#         power = X * (Tf + Tm + Tc)
#         u = pid.update(t, power, desired_power)
#         sim(t, u)
#         times.append(t)
#         powers.append(power)
#         t += sim.d_time

#     # plot the actual power profile over time compared to the desired profile
#     plt.plot(times, powers, label='Actual')
#     desired_powers = [desired_profile(t) for t in times]
#     plt.plot(times, desired_powers, label='Desired')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Power')
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
#     main()

