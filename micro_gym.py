import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from point_kinetics import MicroReactorSimulator
from controllers import PIDController


class kMicroEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    # Reactor parameters
    Sig_x = 2.65e-22
    yi = 0.061
    yx = 0.002
    lamda_x = 2.09e-5
    lamda_I = 2.87e-5
    Sum_f = 0.3358
    l = 1.68e-3
    beta = 0.0048
    betas = np.array([1.42481E-04, 9.24281E-04, 7.79956E-04, 2.06583E-03, 6.71175E-04, 2.17806E-04])
    lambdas = np.array([1.272E-02, 3.174E-02, 1.160E-01, 3.110E-01, 1.400E+00, 3.870E+00])

    cp_f = 977
    cp_m = 1697
    cp_c = 5188.6
    M_f = 2002
    M_m = 11573
    M_c = 500
    mu_f = M_f * cp_f
    mu_m = M_m * cp_m
    mu_c = M_c * cp_c
    f_f = 0.96
    P_0 = 22e6
    Tf0 = 1105
    Tm0 = 1087
    T_in = 864
    T_out = 1106
    Tc0 = (T_in + T_out) / 2
    K_fm = f_f * P_0 / (Tf0 - Tm0)
    K_mc = P_0 / (Tm0 - Tc0)
    M_dot = 1.75E+01
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    G = 3.2e-11
    V = 400 * 200
    Pi = P_0 / (G * Sum_f * V)
    Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
    I0 = yi * Sum_f * Pi / lamda_I

    Rho_d0 = -0.033085599
    Reactivity_per_degree = 26.11e-5
    u0 = 77.56
    energy_per_fission_fuel = 3.2e-11
    core_volume = 400 * 200
    rated_power = 22e6
    Pi = rated_power / (energy_per_fission_fuel * Sum_f * core_volume)
    Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
    I0 = yi * Sum_f * Pi / lamda_I
    Rho_d1 = Rho_d0 + u0 * Reactivity_per_degree

    def __init__(self, dt=0.1, episode_length=200,
                 render_mode=None, run_name=None):
        self.dt = dt
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        self.episode_length = episode_length
        self.run_name = run_name


        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.reset()


    def reset(self, seed=None, options=None):
        super(self.__class__, self).reset(seed=seed)
        self.desired_profile = random_desired_profile()
        initial_power = self.desired_profile(0)
        self.n_r = initial_power / 100
        self.precursor_concentrations = np.array([self.n_r] * 6)
        self.time = 0
        self.X = self.Xe0
        self.I = self.I0
        self.Tf = self.Tf0
        self.Tm = self.Tm0
        self.Tc = self.Tc0
        self.Rho_d1 = 0

        current_power = self.n_r * 100
        next_desired_power = self.desired_profile(self.time + 1)
        observation = {
            "next_desired_power": np.array([next_desired_power]),
            "power": np.array([current_power]),
        }

        self.power_history = [current_power]

        return observation, {}


    def convert_action(self, action):
        """Convert from the -1 to 1 box space to -0.5 to 0.5"""
        # return -0.5 + action / 100.0
        # TODO
        if isinstance(action, float):
            return action / 2
        return action.item() / 2


    def step(self, action):
        if self.time >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        real_action = self.convert_action(action)
        num_steps = int(1 / self.dt)
        for _ in range(num_steps):
            self.reactor_dae(real_action / num_steps)
        self.time += 1

        current_power = self.n_r * 100
        next_desired_power = self.desired_profile(self.time + 1)

        observation = {
            "next_desired_power": np.array([next_desired_power]),
            "power": np.array([current_power]),
        }        
        reward, terminated = self.calc_reward()
        truncated = False
        if self.time >= self.episode_length:
            truncated = True
        info = {}

        return observation, reward, terminated, truncated, info


    def calc_reward(self):
        """Returns reward and whether the episode is terminated."""
        power = self.n_r * 100
        desired_power = self.desired_profile(self.time)
        diff = abs(power - desired_power)
        reward = min(100, 1 / diff)
        terminated = False

        acceptable_error = 10 * np.exp(-self.time / 200)
        print(power, desired_power, acceptable_error)
        if power > 110 or diff > acceptable_error:
            reward = -1000
            terminated = True
 
        return reward, terminated
    

    def render(self):
        # plot the actual power profile over time compared to the desired profile
        plt.clf()
        plt.plot(self.power_history,
                 label='Actual')
        plt.plot([self.desired_profile(t) for t in range(len(self.power_history))],
                 label='Desired')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.legend()
        plt.pause(.001)

        if self.run_name:
            plt.savefig(f'runs/{self.run_name}/{self.time}.png')


    def convert_action_to_gym(self, action):
        """Convert from the -0.5 to 0.5 to the 0 to 100 discrete gym action space"""
        # return round((action + 0.5) * 100.0)
        return action * 2


    def reactor_dae(self, drum_rotation):
        self.Rho_d1 += drum_rotation * self.Reactivity_per_degree

        # ODEs
        rho = self.Rho_d1 + self.alpha_f * (self.Tf - self.Tf0) + self.alpha_c * (self.Tc - self.Tc0) + self.alpha_m * (self.Tm - self.Tm0) - self.Sig_x * (self.X - self.Xe0) / self.Sum_f
        print(rho, self.Rho_d1)

        # Kinetics equations with six-delayed neutron groups        
        d_n_r = (rho - self.beta) * self.n_r / self.l + np.sum(self.betas * self.precursor_concentrations / self.l)
        d_precursor_concentrations = self.lambdas * self.n_r - self.lambdas * self.precursor_concentrations
        
        # Xenon and Iodine dynamics
        d_xenon = self.yx * self.Sum_f * self.Pi + self.lamda_I * self.I - self.Sig_x * self.X * self.Pi - self.lamda_x * self.X
        d_iodine = self.yi * self.Sum_f * self.Pi - self.lamda_I * self.I

        # Thermal–hydraulics model of the reactor core
        d_fuel_temp = self.f_f * self.P_0 / self.mu_f * self.n_r - self.K_fm / self.mu_f * (self.Tf - self.Tc)
        d_moderator_temp = (1 - self.f_f) * self.P_0 / self.mu_m * self.n_r + (self.K_fm * (self.Tf - self.Tm) - self.K_mc * (self.Tm - self.Tc)) / self.mu_m
        d_coolant_temp = self.K_mc * (self.Tm - self.Tc) / self.mu_c - 2 * self.M_dot * self.cp_c * (self.Tc - self.T_in) / self.mu_c

        print(d_n_r, d_precursor_concentrations, d_xenon, d_iodine, d_fuel_temp, d_moderator_temp, d_coolant_temp)

        self.n_r += d_n_r * self.dt
        self.precursor_concentrations += d_precursor_concentrations  * self.dt
        self.X += d_xenon  * self.dt
        self.I += d_iodine * self.dt
        self.Tf += d_fuel_temp * self.dt
        self.Tm += d_moderator_temp  * self.dt
        self.Tc += d_coolant_temp  * self.dt




class MicroEnv(gym.Env):
    # WORK IN PROGRESS
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
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
        """Convert from the -1 to 1 box space to -0.5 to 0.5"""
        # return -0.5 + action / 100.0
        # TODO
        if isinstance(action, float):
            return action / 2
        return action.item() / 2

    def step(self, action):
        if self.time >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        true_action = self.convert_action(action)
        power, _precursors = self.simulator.step(true_action)
        self.time += 1

        # normalize observations between 0 and 1
        # normalized_drum_position = self.simulator.drum_position / 180
        normalized_desired_power = self.desired_profile(self.time + 1) / 100
        normalized_power = power / 100

        observation = {
            "desired_power": np.array([normalized_desired_power]),
            "power": np.array([normalized_power]),
        }

        reward, terminated = self.calc_reward(power)
        truncated = False
        if self.time >= self.episode_length-1:
            truncated = True
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info


    def calc_reward(self, power):
        """Returns reward and whether the episode is terminated."""
        # power = self.n_r * 100
        desired_power = self.desired_profile(self.time)
        diff = power - desired_power
        denominator = max(0.01, abs(diff))
        reward = 1 / denominator
        terminated = False

        acceptable_over = 10 * np.exp(-self.time / 200)
        acceptable_under = 5 * np.exp(-self.time / 200)
        if diff > acceptable_over or -diff > acceptable_under:
            reward = -1000
            terminated = True
 
        return reward, terminated


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator = MicroReactorSimulator(self.num_drums, self.d_time)
        # self.fig = plt.figure(figsize=(5,5), dpi=100)
        self.time = 0
        power = 100
        self.desired_profile = random_desired_profile()
        # self.desired_profile = random_desired_profile(hardcoded=True)
        if self.render_mode == "human":
            plt.ion()
            self.render()

        normalized_desired_power = self.desired_profile(self.time + 1) / 100
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
                 [self.desired_profile(t) for t in self.simulator.time_history],
                 label='Desired')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.legend()
        plt.pause(.001)

        if self.run_name:
            plt.savefig(f'runs/{self.run_name}/{self.time}.png')
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
    # [100, 80, 70, 40, 60, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
    # [100, 40, 70, 40, 80, 40],
]

def random_desired_profile(length=200, hardcoded=False):
    num_cutoffs = np.random.randint(3, 7)
    cutoffs = [int(x) for x in np.linspace(length/num_cutoffs, length, num_cutoffs, endpoint=False)]
    cutoffs += np.random.randint(-length//num_cutoffs//4, length//num_cutoffs//4, size=num_cutoffs)
    np.clip(cutoffs, 0, length, out=cutoffs)
    cutoffs.sort()
    values = [100] + [100 * round(x, 1) for x in np.random.uniform(0.4, 1, size=num_cutoffs)]

    # # WARNING: hardcoded values
    if hardcoded:
        cutoffs = np.array(hardcoded_cutoffs[0])
        values = hardcoded_values[0]

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
        action = 0.
        while not done:
            # gym_action = env.action_space.sample()
            gym_action = env.convert_action_to_gym(action)
            obs, _, terminated, truncated, _ = env.step(gym_action)
            action = pid.update(env.time, obs["power"], env.desired_profile(env.time))
            if terminated or truncated:
                done = True


if __name__ == '__main__':
    main()
