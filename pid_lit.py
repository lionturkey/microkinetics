import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path


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

    def __init__(self, Kp=10, Ki=0, Kd=.8, Kaw=0.3, T_C=0.2, max_rate=0.5, multiplier=1.0):
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
        self.time = 0
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
        self.power_history.append(self.power)
        self.time += step_size
        self.time_history.append(self.time)
        return self.power, self.precursors

    def zero_reactivity(self):
        self.reactivity_inserted = 0



class MicroEnv(gym.Env):
    # WORK IN PROGRESS
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, num_drums=8, d_time=0.1, episode_length=200, render_mode=None):
        self.num_drums = num_drums
        self.d_time = d_time
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        self.episode_length = episode_length

        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Dict({
            "angle": gym.spaces.Discrete(181),
            "power": gym.spaces.Discrete(181),
        })

        self.reset()
    
    def convert_action(self, action):
        """Convert from the 0 to 10 discrete gym action space to each tenth from -0.5 to 0.5"""
        return -0.5 + action / 10.0

    def step(self, action):
        if self.t >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        true_action = self.convert_action(action)
        power, _precursors = self.simulator.step(true_action)
        self.t += 1

        observation = {
            "angle": self.simulator.drum_position,
            "power": power,
        }

        reward = self.calc_reward(power, true_action)

        if self.t >= self.episode_length:
            truncated = True
        else:
            truncated = False

        info = {
            "actions": true_action,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, False, truncated, info

    def calc_reward(self, power, true_action):
        return 1 / ((power - self.profile(self.t)) + abs(true_action))

    def reset(self):
        self.simulator = MicroReactorSimulator(self.num_drums, self.d_time)
        # self.fig = plt.figure(figsize=(5,5), dpi=100)
        self.t = 0
        self.power = 100
        self.profile = random_desired_profile()
        if self.render_mode == "human":
            plt.ion()
            self.render()

    def render(self):
        # plot the actual power profile over time compared to the desired profile
        # plt.ion()
        # self.fig.clf()
        plt.clf()
        plt.plot(self.simulator.time_history, self.simulator.power_history, label='Actual')
        plt.plot(self.simulator.time_history, [self.profile(t) for t in self.simulator.time_history], label='Desired')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.legend()
        plt.pause(.001)
        plt.savefig(f'runs/{self.t}.png')

        # ax = self.fig.gca()
        # ax.plot(self.simulator.time_history, self.simulator.power_history, label='Actual')
        # desired_powers = [self.profile(t) for t in self.simulator.time_history]
        # ax.plot(self.simulator.time_history, desired_powers, label='Desired')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Power')
        # ax.legend()
        # # ax.set_axis_off()
        # # self.fig.show()
        # plt.pause(0.01)

        # image_from_plot = np.frombuffer(self.fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        # # image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # if self.render_mode == "human":
        #     return image_from_plot
        # elif self.render_mode == "rgb_array":
        #     return image_from_plot

    def close(self):
        pass

    def seed(self):
        pass

def random_desired_profile():
    cutoffs = sorted(np.random.randint(0, 200, size=np.random.randint(3, 7)))
    values = [100 * round(x, 1) for x in np.random.uniform(0.4, 1, size=len(cutoffs) + 1)]

    def desired_profile(t):
        for i, cutoff in enumerate(cutoffs):
            if t < cutoff:
                return values[i]
        return values[-1]

    return desired_profile

def convert_action_to_gym(action):
    """Convert from the -0.5 to 0.5 to the 0 to 10 discrete gym action space"""
    return round((action + 0.5) * 10.0)


def main():
    # create a microreactor simulator and a PID controller
    env = MicroEnv(render_mode="human")
    pid = PIDController()

    for _ in range(1):
        env.reset()
        done = False
        action = 0
        while not done:
            # gym_action = env.action_space.sample()
            gym_action = convert_action_to_gym(action)
            obs, _, terminated, truncated, _ = env.step(gym_action)
            action = pid.update(env.t, obs["power"], env.profile(env.t))
            if terminated or truncated:
                done = True


if __name__ == '__main__':
    main()
