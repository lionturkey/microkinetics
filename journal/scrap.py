from envs import HolosPK
import numpy as np
from scipy.integrate import solve_ivp


pke = HolosPK()
y0 = pke.get_initial_conditions()
t = [0,1]
# t_eval = np.linspace(0,1,100)
drum_angles = [77.8]*8
drum_action = [5]*8
drum_forcers = pke.drum_forcing(drum_angles, drum_action)
sol = solve_ivp(pke.reactor_dae, t, y0, args=drum_forcers)
print(len(sol.t))
print(sol.t)
print(sol.y[0])
print(sol.y[-1])
print(sol.y[:,-1])









