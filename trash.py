from sympy import symbols, Eq, solve

f_f = 0.96  # q in paper
P_0 = 22e6 # power
cp_f = 977  # specific heat of fuel
cp_m = 1697  # specific heat of moderator
cp_c = 5188.6  # specific heat of coolant
M_f = 2002  # mass of fuel
M_m = 11573  # mass of moderator
M_c = 500  # mass of coolant
mu_f = M_f * cp_f  
mu_m = M_m * cp_m
mu_c = M_c * cp_c
n_r = 1
# Tf0 = 895
Tf0 = 832.4 # from paper
Tm0 = 893
Tc0 = 888
K_mc = P_0 / (Tm0 - Tc0)
K_fm = f_f * P_0 / (Tf0 - Tm0)

# K_mc = 2.16E+0
# K_fm = 1.17E+06

T_in = 864
M_dot = 1.75E+01
Tf = Tf0
Tm = Tm0
# Tc = Tc0
# d_fuel_temp = f_f * P_0 / mu_f * n_r - K_fm / mu_f * (Tf - Tc)
# using equation solver
Tc = symbols('Tc')
d_fuel_temp_eq = Eq(f_f * P_0 / mu_f * n_r - K_fm / mu_f * (Tf - Tc), 0)
Tc_solution = solve(d_fuel_temp_eq, Tc)
print(Tc_solution)

d_moderator_temp = (1 - f_f) * P_0 / mu_m * n_r + (K_fm * (Tf - Tm) - K_mc * (Tm - Tc)) / mu_m

d_coolant_temp = K_mc * (Tm - Tc) / mu_c - 2 * M_dot * cp_c * (Tc - T_in) / mu_c


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
# Tf0 = 1105 # sooyoung's code
# Tf0 = 900 # kamal's code
# Tf0 = 832.4  # MPACT paper
Tf0 = 895
# Tm0 = 1087 # sooyoung's code
# Tm0 = 898 # kamal's code
# Tm0 = 820 # MPACT paper
Tm0 = 893
T_in = 864  # sooyoung's code
# T_in = 590   # MPACT paper
T_out = 1106  # sooyoung's code
# T_out = 849.1  # MPACT paper
# Tc0 = (T_in + T_out) / 2
# Tc0 = 888 # kamal's code
Tc0 = 888
K_fm = f_f * P_0 / (Tf0 - Tm0)
K_mc = P_0 / (Tm0 - Tc0)
M_dot = 1.75E+01
alpha_f = -2.875e-5
alpha_m = -3.696e-5
alpha_c = 0.0


Tf0 = 832.4  # K
nr = 1
K_fm = 1.17e6  # W/K
K_mc = 2.16e5  # W/K

Tc0 = Tf0 - (f_f*P_0*nr/K_fm)
Tm0 = ((1-f_f)*P_0*nr + Tf0*K_fm + Tc0*K_mc)/(K_fm + K_mc)
Tin = Tc0 - (K_mc*(Tm0-Tc0) / (2*M_dot*cp_c))



if __name__ == '__main__':
    print(f'Tf0: {Tf0}, Tm0: {Tm0}, Tc0: {Tc0}, Tin: {Tin}')

