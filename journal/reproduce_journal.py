import stable_baselines3 as sb3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import envs
import microutils
from scipy.interpolate import interp1d


def main():
    # create interpolated power profiles to match the Holos benchmark
    training_profile = interp1d([  0,  20, 30, 35, 60, 100, 120, 125, 140, 160, 180, 200], # times (s)
                                [100, 100, 90, 90, 55,  55,  65,  65,  80,  80,  95,  95]) # power (SPU)


    ##################
    # PID Controller #
    ##################
    # start with the PID benchmark, creating a run folder
    run_folder = Path.cwd() / 'runs' / 'pid_train'
    run_folder.mkdir(exist_ok=True, parents=True)

    # run the PID loop
    env = envs.HolosSingle(profile=training_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.pid_loop(env)


if __name__ == '__main__':
    main()