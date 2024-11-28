import numpy as np


class PIDController:
    """
    A PID controller.
    Attributes:
        Kp: the proportional gain
        Ki: the integral gain
        Kd: the derivative gain
        Kaw: the anti windup gain
        T_C: the PID parameter
        max_rate: the maximum rate of change of the control signal
        mutiplier: the multiplier for the gains based on the number of drums
    """

    # def __init__(self, Kp=0.3, Ki=0.15, Kd=.1, Kaw=1, T_C=0.2, max_rate=1, multiplier=1):
    def __init__(self, Kp=0.1, Ki=0, Kd=0, Kaw=1, T_C=0.2, max_rate=1, multiplier=1):
        self.Kp = Kp * multiplier
        self.Ki = Ki * multiplier
        self.Kd = Kd * multiplier
        self.Kaw = Kaw * multiplier
        self.T_C = T_C
        self.max_rate = max_rate
        self.integral = 0.0
        self.err_prev = 0.0
        self.t_prev = 0.0
        self.deriv_prev = 0.0

    def update(self, t, measurement, setpoint):
        """
        Update the PID controller.
        Returns:
            the new control signal
        """
        err = setpoint - measurement
        del_t = t - self.t_prev
        self.t_prev = t
        self.integral += self.Ki * err * del_t
        self.deriv_prev = (err - self.err_prev + self.T_C * self.deriv_prev) / (del_t + self.T_C)
        self.err_prev = err
        command = self.Kp * err + self.integral + self.Kd * self.deriv_prev
        # print(f'command {command}')
        command_sat = np.clip(command, -self.max_rate, self.max_rate)
        return command_sat[0]