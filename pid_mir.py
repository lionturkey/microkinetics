import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# %% Part 1: User parameters for the simulation
# you can change the parameters below
# ---------------------------------------------

# choose number of drums to move during the simulation, pick from 8, 4, 2, 1
num_drums = 8
# Time parameters
dt = 0.1  # time step
T = 2000  # total simultion time

time_point = np.array([0, 20, 30, 50, 60, 80, 90, 110, 130, 200]) * 10
pow_point = np.array([1, 1, 0.4, 0.4, 1, 1, 0.4, 0.4, 1, 1])

# note that for the one drum scenario it is hard to reduce the power to 40%
#   without violating the speed limit
# you can try this scenario instead where one drum can reduce the power to 80%
# pow_point = np.array([1, 1, 0.8, 0.8, 1, 1, 0.8, 0.8, 1, 1])

# --you can try other load following scenarios as well
# pow_point = np.array([1, 0.8, 0.6, 0.6, 0.8, 0.8, 0.9, 1, 1, 1])
# pow_point = np.array([1, 1, 0.8, 0.8, 1, 1, 0.4, 0.4, 1, 1])
# pow_point = np.array([1, 0.8, 0.6, 0.4, 0.6, 0.8, 1.0, 0.5, 1, 1])

# %% Part 2: Design-specific parameters for the nuclear reactor
# leave this block as is
# ----------------------------------
time = np.arange(0, T + dt, dt)  # time span
nt = len(time)  # number of time steps
ref = np.zeros(nt)

# ---
# loop to update the reference power history and match the time and power
#   vectors
# ---
ref_old = pow_point[0]
for it in range(nt):
    if it > 0:
        time[it] = time[it - 1] + dt
        ref_old = ref[it - 1]
    ref[it] = ref_old
    for ii in range(len(time_point) - 1):
        if time_point[ii] <= time[it] <= time_point[ii + 1]:
            frac1 = (time_point[ii + 1] - time[it]) / (
                time_point[ii + 1] - time_point[ii]
            )
            frac2 = 1.0 - frac1
            ref[it] = frac1 * pow_point[ii] + frac2 * pow_point[ii + 1]
            break

# ---
# parameters for each drum rotation
# ---
if num_drums == 8:
    Rho_d0 = -0.033085599
    Reactivity_per_degree = 26.11e-5
    u0 = 77.56
elif num_drums == 4:
    Rho_d0 = -0.033085599
    Reactivity_per_degree = 16.11e-5
    u0 = 125.56
elif num_drums == 2:
    Rho_d0 = -0.033085599 + 0.0071
    Reactivity_per_degree = 7.33e-5
    u0 = 177.84
elif num_drums == 1:
    Rho_d0 = -0.033085599 + 0.0071 + 0.0082
    Reactivity_per_degree = 2.77e-5
    u0 = 179.0
else:
    Rho_d0 = -0.033085599
    Reactivity_per_degree = 26.11e-5
    u0 = 77.56

# ---
# Don't change (max reactivity worth for all 8 drums, about 26 pcm/deg of
#   movement)
max_Reactivity_per_degree = 26.11e-5

Kp = 1  # PID Proportional gain
Ki = 1.5  # PID Integral gain
Kaw = 0.3  # PID Anti-windup gain
Kd = 0.001  # PID Derivative gain
T_C = 0.2  # PID parameter
max_val = 180  # max drum position in degree (fully reactivity withdrawal)
min_val = 0  # minimum drum in degree (full reactivity insertion)
max_rate_orig = 0.5  # maximum drum speed allowed (rate of rotation in deg/sec)

# Adjust gains based on number of drums (from reactivity per deg) --> maximum is
#   26.11e-5
Kp *= max_Reactivity_per_degree / Reactivity_per_degree
Ki *= max_Reactivity_per_degree / Reactivity_per_degree
Kaw *= max_Reactivity_per_degree / Reactivity_per_degree
Kd *= max_Reactivity_per_degree / Reactivity_per_degree
max_rate = max_rate_orig * max_Reactivity_per_degree / Reactivity_per_degree

# %% Part 3: PID Control Function
# ----------------------------------


# PID controller
def pid_controller(
    t, measurement, setpoint, Kp, Ki, Kd, Kaw, T_C, max_val, min_val, max_rate,
    init, u0
):
    global integral, err_prev, t_prev, deriv_prev, command_prev, command_sat_prev

    if init:
        integral = u0
        err_prev = 0
        t_prev = 0
        deriv_prev = 0
        command_prev = u0
        command_sat_prev = u0

    err = setpoint - measurement
    T = t - t_prev
    t_prev = t

    integral += Ki * err * T + Kaw * (command_sat_prev - command_prev) * T

    deriv_filt = (err - err_prev + T_C * deriv_prev) / (T + T_C)

    err_prev = err
    deriv_prev = deriv_filt

    command = Kp * err + integral + Kd * deriv_filt
    command_prev = command

    # apply drum position limits
    command_sat = np.clip(command, min_val, max_val)

    # apply drum speed limit
    if command_sat > command_sat_prev + max_rate * T:
        command_sat = command_sat_prev + max_rate * T
    elif command_sat < command_sat_prev - max_rate * T:
        command_sat = command_sat_prev - max_rate * T

    command_sat_prev = command_sat

    return command_sat


# %% Part 4: Point Kinetics Reactor Model (leave as is)
# --------------------------------------------------------


def reactor_dae(x, u, Rho_d0, Reactivity_per_degree):
    Sig_x = 2.65e-22
    yi = 0.061
    yx = 0.002
    lamda_x = 2.09e-5
    lamda_I = 2.87e-5
    Sum_f = 0.3358

    l = 1.68e-3
    beta = 0.0048
    beta_1 = 1.42481e-4
    beta_2 = 9.24281e-4
    beta_3 = 7.79956e-4
    beta_4 = 2.06583e-3
    beta_5 = 6.71175e-4
    beta_6 = 2.17806e-4
    Lamda_1 = 1.272e-2
    Lamda_2 = 3.174e-2
    Lamda_3 = 1.160e-1
    Lamda_4 = 3.110e-1
    Lamda_5 = 1.400
    Lamda_6 = 3.870

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
    M_dot = 1.75e1
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    X0 = 2.35496411413791e10

    n_r, Cr1, Cr2, Cr3, Cr4, Cr5, Cr6, X, I, Tf, Tm, Tc = x

    Rho_d1 = Rho_d0 + u * Reactivity_per_degree

    G = 3.2e-11
    V = 400 * 200
    Pi = P_0 / (G * Sum_f * V)

    dx = np.zeros_like(x)
    rho = (
        Rho_d1
        + alpha_f * (Tf - Tf0)
        + alpha_c * (Tc - Tc0)
        + alpha_m * (Tm - Tm0)
        - Sig_x * (X - X0) / Sum_f
    )

    dx[0] = (
        (rho - beta) / l * n_r
        + beta_1 / l * Cr1
        + beta_2 / l * Cr2
        + beta_3 / l * Cr3
        + beta_4 / l * Cr4
        + beta_5 / l * Cr5
        + beta_6 / l * Cr6
    )
    dx[1] = Lamda_1 * n_r - Lamda_1 * Cr1
    dx[2] = Lamda_2 * n_r - Lamda_2 * Cr2
    dx[3] = Lamda_3 * n_r - Lamda_3 * Cr3
    dx[4] = Lamda_4 * n_r - Lamda_4 * Cr4
    dx[5] = Lamda_5 * n_r - Lamda_5 * Cr5
    dx[6] = Lamda_6 * n_r - Lamda_6 * Cr6

    dx[7] = yx * Sum_f * Pi + lamda_I * I - Sig_x * X * Pi - lamda_x * X
    dx[8] = yi * Sum_f * Pi - lamda_I * I

    dx[9] = f_f * P_0 / mu_f * n_r - K_fm / mu_f * (Tf - Tc)
    dx[10] = (1 - f_f) * P_0 / mu_m * n_r + (K_fm * (Tf - Tm) - K_mc * (Tm - Tc)) / mu_m
    dx[11] = K_mc * (Tm - Tc) / mu_c - 2 * M_dot * cp_c * (Tc - T_in) / mu_c

    return dx


# %% Part 5: Main simulation loop (PID and reactor model interaction)
# --------------------------------------------------------------------

# Initlize the simulation
# initial condition for the reactor states (you may play with it)
x0 = [
    pow_point[0],
    1,
    1,
    1,
    1,
    1,
    1,
    62803189247020.48,  # Xenon concentration
    1018930579495656.25,  # Iodine concentration
    900.42,
    898.28,
    888.261,
]

x = np.zeros((nt, 12))  # system state
x[0, :] = x0  # set initial state
u = np.zeros(nt)  # control signal (drum position)
u[0] = u0  # set the first drum position

# --initiliaze the controller (this is an arbitrary call to get init=True and
#   u0=u0 passed)
pid_controller(
    0, 0, 0, Kp, Ki, Kd, Kaw, T_C, max_val, min_val, max_rate, init=True, u0=u0
)

# --Now start the simualtion loop
for i in range(nt - 1):
    dx = reactor_dae(x[i, :], u[i], Rho_d0, Reactivity_per_degree)
    x[i + 1, :] = x[i, :] + dx * dt
    u[i + 1] = pid_controller(
        time[i],
        x[i + 1, 0],
        ref[i + 1],
        Kp,
        Ki,
        Kd,
        Kaw,
        T_C,
        max_val,
        min_val,
        max_rate,
        False,
        0,
    )

# -- rate of change (Add a zero at the beginning to match the length)
du = np.concatenate(([0], np.diff(u) / dt))

# %% Part 6: Postprocessing
# -----------------------------

# Interpolate ref in case t and time are different
interp_func = interp1d(time, ref, kind="linear")
ref_interpolated = interp_func(time)
r = ref_interpolated.copy()
y = x[:, 0]

# Downsampling
downsample_factor = 10
t_ds = time[::downsample_factor]
r_ds = r[::downsample_factor]
y_ds = y[::downsample_factor]

# Calculate the error
e_ds = r_ds - y_ds

# --Metrics
MAE = np.mean(np.abs(e_ds)) * 100
MXE = np.max(np.abs(e_ds)) * 100
max_position = np.max(u)
min_position = np.min(u)
max_speed = np.max(du)
min_speed = np.min(du)

# --Display report
print("********* Simulation Summary *******")
print(f"Mean Absolute Error in Power (MAE): {MAE:.2}%")
print(f"Max Absolute Error at any point in time (MXE): {MXE:.2f}%")
print(
    f"Min and Max drum position at any point in time: Max: {max_position:.1f}, Min: {min_position:.1f}"
)
print(
    f"Min and Max drum speed at any point in time: Max: {max_speed:.2f} (allowed: {max_rate_orig}), Min: {min_speed:.2f} (allowed: {-max_rate_orig})"
)
print("************************************")

# --Plot results
plt.figure(figsize=(12, 12))

# Power subplot
plt.subplot(3, 1, 1)
plt.plot(time, x[:, 0] * 100, linewidth=2, label="Calculated power")
plt.plot(time, ref_interpolated * 100, "--", linewidth=2, label="Target power")
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Power (%)")
plt.title("PID microreactor control system core power simulation")
plt.ylim([0, 200])
plt.legend()

# Rotation subplot
plt.subplot(3, 1, 2)
plt.plot(time, u, linewidth=2)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Rotation (deg)")
plt.title("Rotation")

if num_drums == 8:
    plt.ylim([65, 85])
elif num_drums == 4:
    plt.ylim([110, 135])
elif num_drums == 2:
    plt.ylim([120, 190])
elif num_drums == 1:
    plt.ylim([120, 185])
else:
    plt.ylim([65, 85])

# Rotation rate of change subplot
plt.subplot(3, 1, 3)
plt.plot(time, du, linewidth=2)
plt.plot(time, [max_rate_orig] * len(time), linewidth=2, label="Max rate")
plt.plot(time, [-max_rate_orig] * len(time), linewidth=2, label="Min rate")
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Rate of change (deg/s)")
plt.title("Rate of change")
plt.legend()
plt.ylim([-1, 1])

# Save the plot as a PNG file
plt.savefig("pid_results.png", dpi=300, bbox_inches="tight")
# plt.show()

print(MAE / 100)
