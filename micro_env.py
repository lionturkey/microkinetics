import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env


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
        return command_sat[0]


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

    def __init__(self, num_drums=8, d_time=0.1, episode_length=200,
                 render_mode=None, run_name=None):
        self.num_drums = num_drums
        self.d_time = d_time
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        self.episode_length = episode_length
        self.run_name = run_name

        self.action_space = gym.spaces.Discrete(101)
        self.observation_space = gym.spaces.Dict({
            "angle": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        self.reset()
    
    def convert_action(self, action):
        """Convert from the 0 to 10 discrete gym action space to each tenth from -0.5 to 0.5"""
        return -0.5 + action / 100.0

    def step(self, action):
        if self.t >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        true_action = self.convert_action(action)
        power, _precursors = self.simulator.step(true_action)
        self.t += 1

        # normalize observations between 0 and 1
        normalized_drum_position = self.simulator.drum_position / 180
        normalized_power = power / 100

        observation = {
            "angle": np.array([normalized_drum_position]),
            "power": np.array([normalized_power]),
        }

        reward = self.calc_reward(power, true_action)

        if self.t >= self.episode_length:
            truncated = True
        else:
            truncated = False
        
        if power > 105 or abs(power - self.profile(self.t)) > 10:
            reward = -100
            terminated = True
        else:
            terminated = False

        info = {
            "actions": true_action,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def calc_reward(self, power, true_action):
        diff = min(100, abs(power - self.profile(self.t)))
        if diff < 5:
            return 1 / diff
        else:
            return -diff


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator = MicroReactorSimulator(self.num_drums, self.d_time)
        # self.fig = plt.figure(figsize=(5,5), dpi=100)
        self.t = 0
        power = 100
        self.profile = random_desired_profile()
        if self.render_mode == "human":
            plt.ion()
            self.render()

        observation = {
            "angle": np.array([self.simulator.drum_position]),
            "power": np.array([power]),
        }
        return observation, {}

    def render(self):
        # plot the actual power profile over time compared to the desired profile
        # plt.ion()
        # self.fig.clf()
        plt.clf()
        plt.plot(self.simulator.time_history, self.simulator.power_history,
                 label='Actual')
        plt.plot(self.simulator.time_history,
                 [self.profile(t) for t in self.simulator.time_history],
                 label='Desired')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.legend()
        plt.pause(.001)

        if self.run_name:
            plt.savefig(f'runs/{self.run_name}/{self.t}.png')
        # if self.render_mode == "human":
        #     return image_from_plot
        # elif self.render_mode == "rgb_array":
        #     return image_from_plot

    def convert_action_to_gym(self, action):
        """Convert from the -0.5 to 0.5 to the 0 to 100 discrete gym action space"""
        return round((action + 0.5) * 100.0)

    def close(self):
        pass

    def seed(self):
        pass

def random_desired_profile(length=200):
    num_cutoffs = np.random.randint(3, 7)
    cutoffs = [int(x) for x in np.linspace(length/num_cutoffs, length, num_cutoffs, endpoint=False)]
    cutoffs += np.random.randint(-length//num_cutoffs//4, length//num_cutoffs//4, size=num_cutoffs)
    np.clip(cutoffs, 0, length, out=cutoffs)
    cutoffs.sort()
    values = [100] + [100 * round(x, 1) for x in np.random.uniform(0.4, 1, size=num_cutoffs)]

    def desired_profile(t):
        for i, cutoff in enumerate(cutoffs):
            if abs(t - cutoff) < 5: # WARNING: magic number ramp window of 10
                # use point slope form: y = m(x-x0) + y0 at cutoff
                power = (((values[i+1] - values[i]) / 10) *
                         (t - cutoff) + (values[i] + values[i+1]) / 2)
                return power
            elif t < cutoff:
                return values[i]
        return values[-1]

    return desired_profile




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
            gym_action = env.convert_action_to_gym(action)
            obs, _, terminated, truncated, _ = env.step(gym_action)
            action = pid.update(env.t, obs["power"], env.profile(env.t))
            if terminated or truncated:
                done = True


    # # RL training and testing loop using the microreactor environment and ppo

    # vec_env = make_vec_env(MicroEnv, n_envs=6, env_kwargs={'render_mode': None})
    # model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1)
    # model.learn(total_timesteps=10000000)

    # model.save("ppo_microreactor_10mil")

    # # Test the trained agent
    
    # env = MicroEnv(render_mode="human")
    # obs, _ = env.reset()
    # rewards = []
    
    # done = False
    # while not done:
    #     # gym_action = env.action_space.sample()
    #     gym_action, _states = model.predict(obs)
    #     # gym_action = convert_action_to_gym(action)
    #     obs, reward, terminated, truncated, _ = env.step(gym_action)
    #     rewards.append(reward)
    #     if terminated or truncated:
    #         done = True
    #     vec_env.render(mode="human")

    # print(sum(rewards))


if __name__ == '__main__':
    main()
