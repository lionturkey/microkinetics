import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# def calc_metrics():
#     pass


class PIDController:
    """
    A PID controller.
    Attributes:
        Kp: the proportional gain
        Ki: the integral gain
        Kd: the derivative gain
        max_rate: the maximum rate of change of the control signal
        mutiplier: the multiplier for the gains based on the number of drums
    """

    # def __init__(self, Kp=0.3, Ki=0.15, Kd=.1, max_rate=1, multiplier=1):
    def __init__(self, Kp=0.2, Ki=0, Kd=.3, max_rate=1, multiplier=1):
        self.Kp = Kp * multiplier
        self.Ki = Ki * multiplier
        self.Kd = Kd * multiplier
        self.max_rate = max_rate
        self.integral = 0.0
        self.err_prev = 0.0
        self.t_prev = 0.0
        self.deriv_prev = 0.0

    def update(self, measurement, setpoint):
        """
        Update the PID controller.
        Returns:
            the new control signal
        """
        err = setpoint - measurement
        del_t = 1
        self.integral += self.Ki * err * del_t
        self.deriv_prev = (err - self.err_prev) / (del_t)
        self.err_prev = err
        command = self.Kp * err + self.integral + self.Kd * self.deriv_prev
        # print(f'command {command}')
        command_sat = np.clip([command], -self.max_rate, self.max_rate)  # array
        return command_sat


def pid_loop(single_env):
    controller = PIDController()
    obs, _ = single_env.reset()
    done = False
    while not done:
        action = controller.update(obs["power"]*100,
                                   single_env.profile(single_env.time+1))
        obs, _, terminated, truncated, _ = single_env.step(action)
        if terminated or truncated:
            done = True
    single_env.render()


def control_loop(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
    env.render()


def marl_control_loop(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
    env.render()


def find_latest_file(folder_path: Path, pattern: str='*') -> Path:
    assert folder_path.exists()
    assert any(folder_path.glob(pattern)), f"No files match pattern '{pattern}' in folder '{folder_path}'"
    latest_file = sorted(folder_path.glob(pattern), key=os.path.getmtime, reverse=True)[0]
    assert latest_file.is_file(), f"Latest file '{latest_file}' is not a file"
    return latest_file


def load_history(history_path: Path):
    assert history_path.exists()
    history = pd.read_csv(history_path)
    assert history['desired_power'][0] == 1, 'steady state initial power value should be 100'
    assert history['drum_8'][0] == 77.8, 'steady state initial drum angle should be 77.8'
    return history


def calc_metrics(history: pd.DataFrame):
    assert history['time'][1] - history['time'][0] == 1, 'metric calculations assume 1 second timesteps'
    error = history['desired_power'] - history['power']
    absolute_error = np.abs(error)
    mean_absolute_error = np.mean(absolute_error)
    integral_absolute_error = np.sum(absolute_error)
    drum_angles = history[['drum_1', 'drum_2', 'drum_3', 'drum_4', 'drum_5', 'drum_6', 'drum_7', 'drum_8']]
    drum_speeds = np.diff(drum_angles, axis=0)
    absolute_drum_speeds = np.abs(drum_speeds)
    control_effort = np.sum(absolute_drum_speeds)
    return mean_absolute_error, integral_absolute_error, control_effort


def plot_history(history: pd.DataFrame):
    plt.clf()
    plt.plot(history['time'], history['actual_power'])
    plt.plot(history['time'], history['desired_power'])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (SPU)')
    plt.title('Power vs. Time')
    plt.show()

# if run_name is not None:
#     self.run_folder = Path.cwd() / 'runs' / run_name
#     self.run_folder.mkdir(parents=True, exist_ok=True)