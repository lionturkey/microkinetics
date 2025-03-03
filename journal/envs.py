from matplotlib.scale import InvertedSymmetricalLogTransform
import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import microutils


class HolosPK:
    """Class representing the point kinetics equations
        for the Holos-Quad microreactor"""
    ##############################
    # Default reactor parameters #
    ##############################
    # values mostly from table 2 inChoi 2020
    sigma_Xe = 2.65e-22  # xenon micro xsec m^2 according to online sources
    yield_I = 0.061  # yield iodine
    yield_Xe = 0.002  # yield xenon
    lambda_Xe = 2.09e-5  # decay of xenon s^-1
    lambda_I = 2.87e-5  # decay of iodine s^-1
    Sigma_f = 0.1117  # macro xsec fission m^-1, worked out backwards from estimated values in Choi 2020 p. 28 Fig.15c
    therm_n_vel = 2.19e3  # thermal neutron velocity m/s according to Wikipedia on neutron temp (0.25 eV)
    neutron_lifetime = 1.68e-3  # s
    beta = 0.004801
    betas = np.array([1.42481E-04, 9.24281E-04, 7.79956E-04, 2.06583E-03, 6.71175E-04, 2.17806E-04])
    lambdas = np.array([1.272E-02, 3.174E-02, 1.160E-01, 3.110E-01, 1.400E+00, 3.870E+00])
    cp_f = 977  # specific heat of fuel
    cp_m = 1697  # specific heat of moderator
    cp_c = 5188.6  # specific heat of coolant
    M_f = 2002  # mass of fuel
    M_m = 11573  # mass of moderator
    M_c = 500  # mass of coolant
    heat_f = 0.96  # q in paper, fraction of heat deposited in fuel
    Tf0 = 832.4
    Tm0 = 830.22  # MPACT paper
    T_in = 795.47  # computed to make initial conditions at steady state
    T_out = 1106  # sooyoung's code
    Tc0 = 814.35  # computed to make initial conditions at steady state
    K_fm = 1.17e6  # W/K
    K_mc = 2.16e5  # W/K
    M_dot = 17.5  # kg/s
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    n_0 = 2.25e13  # m^-3
    P_r = 22e6  # rated power in Watts
    u0 = 77.8  # degrees, steady state full power drum angle (77.56 earlier)
    rho_max = .00500  # max reactivity per drum (500 pcm)

    def __init__(self):
        # calculate steady state conditions and drum reactivity
        self.rho_ss = self.rho_max * (1 - np.cos(np.deg2rad(self.u0))) / 2
        self.I0 = self.yield_I * self.Sigma_f * self.therm_n_vel * self.n_0 / self.lambda_I
        self.Xe0 = ((self.yield_Xe * self.Sigma_f * self.therm_n_vel * self.n_0
                     + self.lambda_I * self.I0)
                    / (self.lambda_Xe
                       + self.sigma_Xe * self.therm_n_vel * self.n_0))

    def get_initial_conditions(self):
        n_r = 1
        c1, c2, c3, c4, c5, c6 = [n_r] * 6
        Tf = self.Tf0
        Tm = self.Tm0
        Tc = self.Tc0
        Xe = self.Xe0
        I = self.I0

        return [n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, Xe, I]

    def calc_reactivity(self, y, drum_angles):
        _, _, _, _, _, _, _, Tf, Tm, _, Xe, _ = y

        drum_reactivity = np.sum(self.rho_max * (1 - np.cos(np.deg2rad(drum_angles))) / 2 - self.rho_ss)

        rho = (drum_reactivity
               + self.alpha_f * (Tf - self.Tf0)
               + self.alpha_m * (Tm - self.Tm0)
               - self.sigma_Xe * (Xe - self.Xe0) / self.Sigma_f)

        return rho

    def drum_forcing(self, drum_angles, drum_action, time = 1):
        """Create a drum angle forcer for intermediate timesteps during a solve_ivp"""
        drum_forcers = []
        for i, drum_angle in enumerate(drum_angles):
            drum_forcers.append(interp1d([0, time], [drum_angle, drum_angle + drum_action[i]]))

        return drum_forcers

    def reactor_dae(self, t, y, d1, d2, d3, d4, d5, d6, d7, d8):
        # note that d1, ..., d8 are expected to be functions of time
        n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, Xe, I = y
        drum_angles = np.array([d1(t), d2(t), d3(t), d4(t), d5(t), d6(t), d7(t), d8(t)])
        rho = self.calc_reactivity(y, drum_angles)
        precursor_concentrations = np.array([c1, c2, c3, c4, c5, c6])

        # Kinetics equations with six-delayed neutron groups        
        d_n_r = (((rho - self.beta) * n_r
                  + np.sum(self.betas * precursor_concentrations))
                 / self.neutron_lifetime)
        d_c1, d_c2, d_c3, d_c4, d_c5, d_c6 = (self.lambdas * n_r
                                              - self.lambdas * precursor_concentrations)

        # Thermal–hydraulics model of the reactor core
        d_Tf = ((self.heat_f * self.P_r * n_r
                 - self.K_fm * (Tf - Tc))
                / (self.M_f * self.cp_f))
        
        d_Tm = (((1 - self.heat_f) * self.P_r * n_r
                 + self.K_fm * (Tf - Tm)
                 - self.K_mc * (Tm - Tc))
                / (self.M_m * self.cp_m))
        d_Tc = ((self.K_mc * (Tm - Tc)
                 - 2*self.M_dot * self.cp_c * (Tc - self.T_in))
                / (self.M_c * self.cp_c))

        # Xenon and Iodine dynamics
        n_rate_density = self.therm_n_vel * self.n_0 * n_r
        d_I = (self.yield_I * self.Sigma_f * n_rate_density
               - self.lambda_I * I)
        d_Xe = (self.yield_Xe * self.Sigma_f * n_rate_density
                + self.lambda_I * I
                - self.lambda_Xe * Xe
                - self.sigma_Xe * Xe * n_rate_density)

        return [d_n_r, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6, d_Tf, d_Tm, d_Tc, d_Xe, d_I]


class HolosMulti(gym.Env):
    def __init__(self, profile, episode_length, run_path=None,
                 run_mode="train", noise=0.0, debug=False):
        self.profile = profile
        self.episode_length = episode_length
        self.run_path = run_path
        self.run_mode = run_mode
        self.noise = noise
        self.debug = debug

        self.pke = HolosPK()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "drum_angles": gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
            
        self.reset()

    def reset(self, seed=None, options=None):
        super(self.__class__, self).reset(seed=seed)
        current_desired_power = self.profile(0) / 100
        self.time = 0
        self.drum_angles = np.array([77.8]*8)
        self.y = self.pke.get_initial_conditions()
        current_power, *_ = self.y
        self.history = [[self.time, *self.drum_angles, current_desired_power, *self.y]]

        next_desired_power = self.profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        observation = {
            "drum_angles": self.drum_angles / 180,  # convert to 0 to 1 box space
            "next_desired_power": np.array([next_desired_power / 100]),
            "power": np.array([current_power + fuzz]),
        }

        return observation, {'latest': self.history[-1]}

    def gym2real_action(self, gym_action):
        """Convert from the -1 to 1 box space to -0.5 to 0.5"""
        real_action = gym_action.item() / 2
        return real_action

    def real2gym_action(self, real_action):
        """Convert from the real -0.5 to 0.5 to the 0 1 continuous gym action space"""
        gym_action = real_action * 2
        return gym_action

    def step(self, action):
        if self.time >= self.episode_length:
            raise RuntimeError("Episode length exceeded")
        real_action = self.gym2real_action(action)

        drum_forcers = self.drum_forcers(self.drum_angles, real_action)
        sol = solve_ivp(self.pke.reactor_dae, [0, 1], self.y, args=drum_forcers)
        self.y = sol.y[:,-1]
        self.drum_angles += real_action
        self.time += 1

        current_power, *_ = self.y
        self.history.append([self.time, *self.drum_angles, current_desired_power, *self.y])
        next_desired_power = self.profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        observation = {
            "drum_angles": self.drum_angles / 180,  # convert to 0-1 box space
            "next_desired_power": np.array([next_desired_power / 100]),  # convert to 0-1 box space
            "power": np.array([current_power + fuzz]),
        }

        desired_power = self.profile(self.time)
        reward, terminated = self.calc_reward((current_power*100), desired_power)
        truncated = False
        if self.time >= self.episode_length:
            truncated = True
        info = {'latest': self.history[-1]}

        return observation, reward, terminated, truncated, info

    def calc_reward(self, current_power, desired_power):
        """Returns reward and whether the episode is terminated."""
        # First component: give reward to stay in the correct range
        diff = abs(current_power - desired_power)
        reward = 2 - diff

        # give a punish outside bounds if in train mode
        terminated = False
        if self.run_mode == "train" and diff > 5:
            reward -= 100
            terminated = True

        return reward, terminated

    def render(self, mode='human'):
        if self.run_name:
            base_name = (f'runs/{self.run_name}/{self.run_mode}-mode_'
                        f'{self.noise}-noise_{self.profile}-profile_'
                        f'{self.reward_mode}-reward_{self.time}')

            df = pd.DataFrame({
                'desired_power': [100] + [self.desired_profile(t) for t in range(self.episode_length)],
                'actual_power': self.power_history,
                'diff': self.diff_history,
                'action': self.action_history,
                'fuel_temp': self.fuel_temp_history,
                'moderator_temp': self.moderator_temp_history,
                'coolant_temp': self.coolant_temp_history,
                'drum': self.drum_history,
                'Xe': self.Xe_history,
                'I': self.I_history,
            })
            df.to_csv(f'{base_name}.csv', index=False)


class HolosSingle(gym.Env):
    def __init__(self, profile, episode_length, run_path=None,
                 run_mode="train", noise=0.0, debug=False):
        self.env = HolosMulti(profile, episode_length, run_path, run_mode, noise, debug)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "drum_angle": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.env.reset()

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        observation = {
            "drum_angle": np.array(np.mean(obs["drum_angles"])),  # treat as a single drum angle
            "next_desired_power": np.array([obs["next_desired_power"]]),
            "power": np.array([obs["power"]]),
        }
        return observation, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        observation = {
            "drum_angle": np.array(np.mean(obs["drum_angles"])),  # treat as a single drum angle
            "next_desired_power": np.array([obs["next_desired_power"]]),
            "power": np.array([obs["power"]]),
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        self.env.render(mode=mode)


# class HolosMARL(ParallelEnv):