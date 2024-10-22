import numpy as np
import gymnasium as gym

class ReactorGym(gym.Env):
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
    

    def __init__(self, initial_power=100, dt=0.1):
        self.dt = dt

        options = {'initial_power': initial_power}
        self.reset(options=options)


    def reset(self, seed=None, options=None):
        super.reset(seed=seed)
        initial_power = options['initial_power']
        self.n_r = initial_power / 100
        self.precursor_concentrations = np.array([self.n_r] * 6)
        self.time = 0
        self.X = self.Xe0
        self.I = self.I0
        self.Tf = self.Tf0
        self.Tm = self.Tm0
        self.Tc = self.Tc0
        
        return self.n_r

    def step(self, action):
        num_steps = int(1 / self.dt)
        for _ in num_steps:
            self.reactor_dae(action / num_steps)
        
#        state = self.n_r
#        reward = compute_reward()....
        
        return state, reward, terminated, truncated, info


    def reactor_dae(self, drum_rotation):
        self.Rho_d1 += drum_rotation * self.Reactivity_per_degree

        # ODEs
        rho = self.Rho_d1 + self.alpha_f * (self.Tf - self.Tf0) + self.alpha_c * (self.Tc - self.Tc0) + self.alpha_m * (self.Tm - self.Tm0) - self.Sig_x * (self.X - self.Xe0) / self.Sum_f

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
        
        self.n_r += d_n_r * self.dt
        self.precursor_concentrations += d_precursor_concentrations  * self.dt
        self.X += d_xenon  * self.dt
        self.I += d_iodine * self.dt
        self.Tf += d_fuel_temp * self.dt
        self.Tm += d_moderator_temp  * self.dt
        self.Tc += d_coolant_temp  * self.dt

