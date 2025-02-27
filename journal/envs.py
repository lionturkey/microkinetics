import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import microutils


class HolosEnv(gym.Env):
    ######################
    # Reactor parameters #
    ######################
    sigma_Xe = 2.6e-22  # xenon micro xsec m^2 according to random online sources
    yield_I = 0.061  # yield iodine
    yield_Xe = 0.002  # yield xenon
    lambda_Xe = 2.09e-5  # decay of xenon s^-1
    lambda_I = 2.87e-5  # decay of iodine s^-1
    # Kamal used 0.3358, but working backwards from estimated values in Choi 2020 p. 28 Fig.15c:
    Sigma_f = 0.1117  # macro xsec fission m^-1
    therm_n_vel = 2.19e3  # thermal neutron velocity m/s according to Wikipedia on neutron temp lol (0.25 eV)
    neutron_lifetime = 1.68e-3  # s
    beta = 0.0048
    betas = np.array([1.42481E-04, 9.24281E-04, 7.79956E-04, 2.06583E-03, 6.71175E-04, 2.17806E-04])
    lambdas = np.array([1.272E-02, 3.174E-02, 1.160E-01, 3.110E-01, 1.400E+00, 3.870E+00])

    # table 2 in Choi 2020
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
    M_dot = 1.75E+01
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    n_0 = 2.25e13  # m^-3
    P_r = 22e6  # rated power in Watts
    reactivity_per_degree = 26.11e-5
    u0 = 77.56  # degrees, steady state full power drum angle
    I0 = yield_I * Sigma_f * therm_n_vel * n_0 / lambda_I
    Xe0 = ((yield_Xe * Sigma_f * therm_n_vel * n_0
            + lambda_I * I0)
           / (lambda_Xe
              + sigma_Xe * therm_n_vel * n_0))


    def __init__(self, profile, episode_length=200, run_name=None,
                 run_mode="train", noise=0.0, debug=False):
        self.profile = profile
        self.episode_length = episode_length
        self.run_name = run_name
        self.debug = debug
        self.run_mode = run_mode
        self.noise = noise

        self.dt = 0.1  # simulation timestep size in seconds

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "last_action": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
            "next_desired_power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "power": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
            
        self.reset()


    def reset(self, seed=None, options=None):
        super(self.__class__, self).reset(seed=seed)
        # self.desired_profile = random_desired_profile()
        profile = 'train' if self.profile == 'train' else 'test'
        fac = self.episode_length / 200
        self.desired_profile = random_desired_profile(hardcoded=True, profile=profile, fac=fac)
        initial_power = self.desired_profile(0)
        self.n_r = initial_power / 100
        self.precursor_concentrations = np.array([self.n_r] * 6)
        self.time = 0
        self.Xe = self.Xe0
        self.I = self.I0
        self.Tf = self.Tf0
        self.Tm = self.Tm0
        self.Tc = self.Tc0
        self.rho_drum = 0
        self.rho = 0

        self.drum = 77.8

        next_desired_power = self.desired_profile(self.time + 1)
        fuzz = np.random.normal(0, self.noise)
        observation = {
            "last_action": np.array([0]),
            "next_desired_power": np.array([next_desired_power/100]),
            "power": np.array([self.n_r + fuzz]),
        }

        self.power_history = [initial_power]
        self.fuel_temp_history = [self.Tf]
        self.moderator_temp_history = [self.Tm]
        self.coolant_temp_history = [self.Tc]
        self.action_history = [0]
        self.diff_history = [0]
        self.drum_history = [self.drum]
        self.Xe_history = [self.Xe]
        self.I_history = [self.I]

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

        num_steps = int(1 / self.dt)
        for _ in range(num_steps):
            self.reactor_dae(real_action / num_steps, debug=self.debug)
        self.time += 1

        current_power = self.n_r * 100
        current_desired_power = self.desired_profile(self.time)
        next_desired_power = self.desired_profile(self.time + 1)
        this_action = self.convert_action_to_gym(real_action)
        self.drum += real_action
        fuzz = np.random.normal(0, self.noise)

        observation = {
            "last_action": np.array([this_action]),
            "next_desired_power": np.array([next_desired_power/100]),
            "power": np.array([self.n_r + fuzz]),
        }        
        reward, terminated = self.calc_reward(current_power)
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
        self.drum_history.append(self.drum)
        self.Xe_history.append(self.Xe)
        self.I_history.append(self.I)

        return observation, reward, terminated, truncated, info


    def calc_reward(self, power):
        """Returns reward and whether the episode is terminated."""
        # First component: give reward to stay in the correct range
        desired_power = self.desired_profile(self.time)
        diff = abs(power - desired_power)
        reward = 2 - diff

        # Third component: give a punish outside bounds if in train mode
        terminated = False
        if self.run_mode == "train" and diff > 5:
            reward -= 100
            terminated = True

        return reward, terminated


    def render(self):
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


    def convert_action_to_gym(self, action):
        """Convert from the real -0.5 to 0.5 to the 0 1 continuous gym action space"""
        return action * 2


    def reactor_dae(self, drum_rotation):
        self.rho_drum += drum_rotation * self.reactivity_per_degree

        # ODEs
        rho = (self.rho_drum
               + self.alpha_f * (self.Tf - self.Tf0)
               + self.alpha_m * (self.Tm - self.Tm0)
                - self.sigma_Xe * (self.Xe - self.Xe0) / self.Sigma_f)
        self.rho = rho

        # Kinetics equations with six-delayed neutron groups        
        d_n_r = (((rho - self.beta) * self.n_r
                  + np.sum(self.betas * self.precursor_concentrations))
                 / self.neutron_lifetime)
        d_precursor_concentrations = (self.lambdas * self.n_r
                                      - self.lambdas * self.precursor_concentrations)
        
        # Xenon and Iodine dynamics
        n_rate_density = self.therm_n_vel * self.n_0 * self.n_r
        d_iodine = (self.yield_I * self.Sigma_f * n_rate_density
                    - self.lambda_I * self.I)
        d_xenon = (self.yield_Xe * self.Sigma_f * n_rate_density
                   + self.lambda_I * self.I
                   - self.lambda_Xe * self.Xe
                   - self.sigma_Xe * self.Xe * n_rate_density)

        # Thermal–hydraulics model of the reactor core
        d_fuel_temp = ((self.heat_f * self.P_r * self.n_r
                        - self.K_fm * (self.Tf - self.Tc))
                       / (self.M_f * self.cp_f))
        
        d_moderator_temp = (((1 - self.heat_f) * self.P_r * self.n_r
                             + self.K_fm * (self.Tf - self.Tm)
                             - self.K_mc * (self.Tm - self.Tc))
                            / (self.M_m * self.cp_m))
        d_coolant_temp = ((self.K_mc * (self.Tm - self.Tc)
                           - 2*self.M_dot * self.cp_c * (self.Tc - self.T_in))
                          / (self.M_c * self.cp_c))

        if self.debug:
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
        self.Xe += d_xenon  * self.dt
        self.I += d_iodine * self.dt
        self.Tf += d_fuel_temp * self.dt
        self.Tm += d_moderator_temp  * self.dt
        self.Tc += d_coolant_temp  * self.dt


