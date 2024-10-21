import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from micro_reactor_simulator import MicroReactorSimulator


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

        # self.action_space = gym.spaces.Discrete(101)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        self.reset()
    
    def convert_action(self, action):
        """Convert from the 0 to 10 discrete gym action space to each tenth from -0.5 to 0.5"""
        # return -0.5 + action / 100.0
        # TODO
        return action.item() / 2

    def step(self, action):
        if self.t >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        true_action = self.convert_action(action)
        power, _precursors = self.simulator.step(true_action)
        self.t += 1

        # normalize observations between 0 and 1
        # normalized_drum_position = self.simulator.drum_position / 180
        normalized_desired_power = self.profile(self.t) / 100
        normalized_power = power / 100

        observation = {
            "desired_power": np.array([normalized_desired_power]),
            "power": np.array([normalized_power]),
        }

        reward = self.calc_reward(power, true_action)

        if self.t >= self.episode_length:
            truncated = True
        else:
            truncated = False
        
        if power > 105 or abs(power - self.profile(self.t)) > 10:
            reward = -1000
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
        diff = abs(power - self.profile(self.t))
        return min(100, 1 / diff)
        # if diff < 5:
        #     return 1 / diff
        # else:
        #     return -diff


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

        normalized_desired_power = self.profile(self.t) / 100
        normalized_power = power / 100
        observation = {
            "desired_power": np.array([normalized_desired_power]),
            "power": np.array([normalized_power]),
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
        # return round((action + 0.5) * 100.0)
        return action * 2

    def close(self):
        pass

    def seed(self):
        pass

hardcoded_cutoffs = [
    # [10, 50, 70, 150, 165],
    [30, 45, 77, 128, 160],
    # [5, 53, 72, 130, 187],
    # [10, 50, 70, 150, 200],
    # [10, 50, 70, 150, 200],
]

hardcoded_values = [
    [100, 80, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
]

def random_desired_profile(length=200):
    num_cutoffs = np.random.randint(3, 7)
    cutoffs = [int(x) for x in np.linspace(length/num_cutoffs, length, num_cutoffs, endpoint=False)]
    cutoffs += np.random.randint(-length//num_cutoffs//4, length//num_cutoffs//4, size=num_cutoffs)
    np.clip(cutoffs, 0, length, out=cutoffs)
    cutoffs.sort()
    values = [100] + [100 * round(x, 1) for x in np.random.uniform(0.4, 1, size=num_cutoffs)]

    # # WARNING: hardcoded values
    # cutoffs = np.array(hardcoded_cutoffs[0])
    # values = hardcoded_values[0]

    def desired_profile(t):
        for i, cutoff in enumerate(cutoffs):
            if abs(t - cutoff) < 8: # WARNING: magic number ramp window of 10
                # use point slope form: y = m(x-x0) + y0 at cutoff
                power = (((values[i+1] - values[i]) / 16) *
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


if __name__ == '__main__':
    main()
