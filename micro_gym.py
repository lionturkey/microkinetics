import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
import imageio
from collections import deque

from controllers import PIDController


def create_gif(run_name: str, png_folder: Path = (Path.cwd() / 'runs')):
    # Build GIF
    png_list = list(png_folder.glob('*.png'))
    num_list = sorted([int(png.stem) for png in png_list])
    png_list = [(png_folder / f'{i}.png') for i in num_list]
    with imageio.get_writer((png_folder / f'{run_name}.gif'), mode='I') as writer:
        for filepath in png_list[::2]:
            image = imageio.imread(filepath)
            writer.append_data(image)
        for filepath in png_list:
            # delete png
            filepath.unlink()


class MicroEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    # Reactor parameters
    Sig_x = 2.65e-22  # xenon micro xsec
    yi = 0.061  # yield iodine
    yx = 0.002  # yield xenon
    lamda_x = 2.09e-5  # decay of xenon
    lamda_I = 2.87e-5  # decay of iodine
    Sum_f = 0.3358  # macro xsec fission
    l = 1.68e-3  # neutron lifetime
    beta = 0.0048
    betas = np.array([1.42481E-04, 9.24281E-04, 7.79956E-04, 2.06583E-03, 6.71175E-04, 2.17806E-04])
    lambdas = np.array([1.272E-02, 3.174E-02, 1.160E-01, 3.110E-01, 1.400E+00, 3.870E+00])

    # table 2 in 
    cp_f = 977  # specific heat of fuel
    cp_m = 1697  # specific heat of moderator
    cp_c = 5188.6  # specific heat of coolant
    M_f = 2002  # mass of fuel
    M_m = 11573  # mass of moderator
    M_c = 500  # mass of coolant
    mu_f = M_f * cp_f  
    mu_m = M_m * cp_m
    mu_c = M_c * cp_c
    f_f = 0.96  # q in paper
    P_0 = 22e6
    Tf0 = 832.4
    Tm0 = 830.22  # MPACT paper
    T_in = 795.47  # calculated in trash.py...
    T_out = 1106  # sooyoung's code
    Tc0 = 814.35  # calculated in trash.py...
    K_fm = 1.17e6  # W/K
    K_mc = 2.16e5  # W/K
    M_dot = 1.75E+01
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    G = 3.2e-11
    V = 400 * 200
    Pi = P_0 / (G * Sum_f * V)
    Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
    I0 = yi * Sum_f * Pi / lamda_I

    Reactivity_per_degree = 26.11e-5
    u0 = 77.56
    energy_per_fission_fuel = 3.2e-11
    core_volume = 400 * 200
    rated_power = 22e6
    Pi = rated_power / (energy_per_fission_fuel * Sum_f * core_volume)
    Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
    I0 = yi * Sum_f * Pi / lamda_I

    def __init__(self, dt=0.1, episode_length=200,
                 render_mode=None, run_name=None, debug=False,
                 scale_graphs=False, train_mode=True, noise=0.0,
                 profile='train', reward='optimal'):
        self.dt = dt
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        self.episode_length = episode_length
        self.run_name = run_name
        self.debug = debug
        self.scale_graphs = scale_graphs
        self.train_mode = train_mode
        self.noise = noise

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            # "desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "last_action": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        if run_name is not None:
            self.run_folder = Path.cwd() / 'runs' / run_name
            self.run_folder.mkdir(parents=True, exist_ok=True)

        _, self.ax = plt.subplots(5, 1, figsize=(8, 12))
            
        self.reset()


    def reset(self, seed=None, options=None):
        super(self.__class__, self).reset(seed=seed)
        # self.desired_profile = random_desired_profile()
        self.desired_profile = random_desired_profile(hardcoded=True)
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
        self.rho = 0

        self.drum = 77.8

        current_power = self.n_r * 100
        current_desired_power = self.desired_profile(self.time)
        next_desired_power = self.desired_profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        observation = {
            # "desired_power": np.array([current_desired_power/100]),
            "last_action": np.array([0]),
            "next_desired_power": np.array([next_desired_power/100]),
            "power": np.array([current_power/100 + fuzz]),
        }

        self.power_history = [current_power]
        self.fuel_temp_history = [self.Tf]
        self.moderator_temp_history = [self.Tm]
        self.coolant_temp_history = [self.Tc]
        self.action_history = [0]
        self.diff_history = [0]
        self.rho_diff_history = [0]
        self.drum_history = [self.drum]

        return observation, {}


    def convert_action(self, action):
        """Convert from the -1 to 1 box space to -0.5 to 0.5"""
        if isinstance(action, float):
            return action / 2
        return action.item() / 2


    def step(self, action):
        if self.time >= self.episode_length:
            raise RuntimeError("Episode length exceeded")

        real_action = self.convert_action(action)
        # print(f'real_action: {real_action}')
        # if abs(real_action) < 0.005:
        #     real_action = 0

        num_steps = int(1 / self.dt)
        for _ in range(num_steps):
            self.reactor_dae(real_action / num_steps, debug=self.debug)
        self.time += 1

        current_power = self.n_r * 100
        current_desired_power = self.desired_profile(self.time)
        next_desired_power = self.desired_profile(self.time + 1)
        this_action = self.convert_action_to_gym(real_action)
        self.drum += this_action

        fuzz = np.random.normal(0, self.noise)
        observation = {
            # "desired_power": np.array([current_desired_power/100]),
            "last_action": np.array([this_action]),
            "next_desired_power": np.array([next_desired_power/100]),
            "power": np.array([current_power/100 + fuzz]),
        }        
        reward, terminated = self.calc_reward(current_power, real_action)
        truncated = False
        if self.time >= self.episode_length:
            truncated = True
        info = {}
        
        # add to histories
        self.power_history.append(current_power)
        self.fuel_temp_history.append(self.Tf)
        self.moderator_temp_history.append(self.Tm)
        self.coolant_temp_history.append(self.Tc)
        self.action_history.append(real_action)
        self.diff_history.append(current_desired_power - current_power)
        self.rho_diff_history.append(self.Rho_d1 - self.rho)
        self.drum_history.append(self.drum)
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    

    def calc_reward(self, power, action):
        """Returns reward and whether the episode is terminated."""
        # First component: give reward to stay in the correct range
        desired_power = self.desired_profile(self.time)
        diff = abs(power - desired_power)
        reward = 4

        # Second component: give reward to stay close to the desired power
        reward -= diff
        # reward -= 20*(action**2)

        # Third component: give a punish outside bounds
        if self.train_mode:
            acceptable_error = 4
        else:
            acceptable_error = 10
        terminated = False
        if diff > acceptable_error:
            reward = -100
            terminated = True
        return reward, terminated


    def render(self):
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[3].cla()
        self.ax[4].cla()
        
        self.ax[0].plot(self.power_history, label='Actual')
        self.ax[0].plot([self.desired_profile(t) for t in range(self.episode_length)], label='Desired', alpha=.5)
        self.ax[0].set_xlabel('Time (s)')
        self.ax[0].set_ylabel('Power (SPU)')
        self.ax[0].legend()
        self.ax[0].set_ylim(0, 110)
        self.ax[0].set_xlim(0, self.episode_length)

        self.ax[1].plot(self.fuel_temp_history, label='Fuel')
        self.ax[1].plot(self.moderator_temp_history, label='Moderator')
        self.ax[1].plot(self.coolant_temp_history, label='Coolant')
        self.ax[1].set_xlabel('Time (s)')
        self.ax[1].set_ylabel('Temperature (°C)')
        self.ax[1].legend()
        self.ax[1].set_xlim(0, self.episode_length)

        self.ax[2].plot(self.action_history)
        self.ax[2].hlines(0, 0, self.episode_length, color='k', linestyle='dashed', alpha=.5)
        self.ax[2].set_xlabel('Time (s)')
        self.ax[2].set_ylabel('Action (°/s)')
        if self.scale_graphs:
            bound = np.max(np.abs(self.action_history)) * 1.1
        else:
            bound = 0.41
        self.ax[2].set_ylim(-bound, bound)
        self.ax[2].set_xlim(0, self.episode_length)

        self.ax[3].plot(self.diff_history)
        self.ax[3].hlines(0, 0, self.episode_length, color='k', linestyle='dashed', alpha=.5)
        self.ax[3].set_xlabel('Time (s)')
        self.ax[3].set_ylabel('desired - actual power (SPU)')
        if self.scale_graphs:
            bound = np.max(np.abs(self.diff_history)) * 1.1
        else:
            bound = 5
        self.ax[3].set_ylim(-bound, bound)
        self.ax[3].set_xlim(0, self.episode_length)

        self.ax[4].plot(self.drum_history)
        self.ax[4].set_xlabel('Time (s)')
        self.ax[4].set_ylabel('drum rotation (°)')
        self.ax[4].set_xlim(0, self.episode_length)

        plt.tight_layout()
        # plt.pause(.001)  # a vestige from the video era
        if self.run_name:
            plt.savefig(f'runs/{self.run_name}/{self.time}.png')


    def convert_action_to_gym(self, action):
        """Convert from the -0.5 to 0.5 to the 0 to 100 discrete gym action space"""
        return action * 2


    def reactor_dae(self, drum_rotation, debug=False):
        self.Rho_d1 += drum_rotation * self.Reactivity_per_degree

        # ODEs
        rho = self.Rho_d1 + self.alpha_f * (self.Tf - self.Tf0) + self.alpha_m * (self.Tm - self.Tm0)
        self.rho = rho
        # - self.Sig_x * (self.X - self.Xe0) / self.Sum_f
        # rho = self.Rho_d1

        # Kinetics equations with six-delayed neutron groups        
        d_n_r = (rho - self.beta) * self.n_r / self.l + np.sum(self.betas * self.precursor_concentrations / self.l)
        d_precursor_concentrations = self.lambdas * self.n_r - self.lambdas * self.precursor_concentrations
        
        # Xenon and Iodine dynamics
        # d_xenon = self.yx * self.Sum_f * self.Pi + self.lamda_I * self.I - self.Sig_x * self.X * self.Pi - self.lamda_x * self.X
        # d_iodine = self.yi * self.Sum_f * self.Pi - self.lamda_I * self.I

        # Thermal–hydraulics model of the reactor core
        d_fuel_temp = self.f_f * self.P_0 / self.mu_f * self.n_r - self.K_fm / self.mu_f * (self.Tf - self.Tc)
        d_moderator_temp = (1 - self.f_f) * self.P_0 / self.mu_m * self.n_r + (self.K_fm * (self.Tf - self.Tm) - self.K_mc * (self.Tm - self.Tc)) / self.mu_m
        d_coolant_temp = self.K_mc * (self.Tm - self.Tc) / self.mu_c - 2 * self.M_dot * self.cp_c * (self.Tc - self.T_in) / self.mu_c


        if debug:
            print('##########################')
            print('######next dae iter#######')
            print('##########################')
            print(f'd_n_r {d_n_r}')
            print(f'd_precursor_concentrations {d_precursor_concentrations}')
            print(f'd_fuel_temp {d_fuel_temp}')
            print(f'd_moderator_temp {d_moderator_temp}')
            print(f'd_coolant_temp {d_coolant_temp}')
            print(f'power {self.power_history[-1]}')

        self.n_r += d_n_r * self.dt
        self.precursor_concentrations += d_precursor_concentrations  * self.dt
        # self.X += d_xenon  * self.dt
        # self.I += d_iodine * self.dt
        self.Tf += d_fuel_temp * self.dt
        self.Tm += d_moderator_temp  * self.dt
        self.Tc += d_coolant_temp  * self.dt


fac = 1
hardcoded_cutoffs = [
    # [10, 50, 80, 130, 165],
    [30, 45, 77, 128, 160],
    # [10, 53, 72, 130, 187],
    # [10, 50, 70, 150, 200],
    # [10, 50, 70, 150, 200],
]

hardcoded_cutoffs = [[x * fac for x in hardcoded_cutoffs[0]]]
hardcoded_values = [
    # [100, 80, 70, 40, 80, 40],
    # [100, 80, 70, 40, 70, 40],
    [100, 80, 70, 50, 65, 90],
    # [100, 80, 95, 70, 50, 65],
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
        flag_th = False
        # flag_th = True
        if not flag_th:
            ramp_length = 8
            if t < 0:
                return 100
            for i, cutoff in enumerate(cutoffs):
                if abs(t - cutoff) < ramp_length: # WARNING: ramp window of 8
                    # use point slope form: y = m(x-x0) + y0 at cutoff
                    power = (((values[i+1] - values[i]) / (2*ramp_length)) *
                            (t - cutoff) + (values[i] + values[i+1]) / 2)
                    return power
                elif t < cutoff:
                    return values[i]
            return values[-1]
        else:
            profile_values = -40 * np.tanh(0.05 * (t - 100)) + 60  # Scaling down t and shifting for variety
            return profile_values

    return desired_profile


def main():
    # create a microreactor simulator and a PID controller
    env = MicroEnv(run_name='ktest')
    pid = PIDController()

    for _ in range(1):
        env.reset()
        done = False
        action = 0.
        while not done:
            # gym_action = env.action_space.sample()
            gym_action = env.convert_action_to_gym(action)
            # print(f'action: {gym_action}')
            obs, _, terminated, truncated, _ = env.step(gym_action)
            # print(f'obs: {obs}')
            action = pid.update(env.time, obs["power"]*100, env.desired_profile(env.time+1))
            if terminated or truncated:
                done = True
    env.render()
    # create_gif('ktest', (Path.cwd() / 'runs' / 'ktest'))


if __name__ == '__main__':
    main()
