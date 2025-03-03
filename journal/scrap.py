from envs import HolosPK
import numpy as np
from scipy.integrate import solve_ivp


pke = HolosPK()
y0 = pke.get_initial_conditions()
t = [0,10000]
drum_angles = [76]*8
sol = solve_ivp(pke.reactor_dae, t, y0, args=drum_angles)
print(len(sol.t))
print(sol.t)
print(sol.y[0])
print(sol.y[-1])









