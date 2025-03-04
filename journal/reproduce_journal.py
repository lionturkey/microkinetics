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
    testing_profile = interp1d([  0,  10, 70, 100, 115, 125, 150, 180, 200], # times (s)
                               [100, 100, 45, 45,   65,  65,  50,  80,  80]) # power (SPU)

    ##################
    # PID Controller #
    ##################
    # start with the PID benchmark, creating a run folder
    run_folder = Path.cwd() / 'runs' / 'pid_train'
    run_folder.mkdir(exist_ok=True, parents=True)

    # run the PID loop
    env = envs.HolosSingle(profile=training_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.pid_loop(env)  # note: this will also create a run_history.csv in the run folder

    ####################
    # Single Action RL #
    ####################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'single_action_rl'
    run_folder.mkdir(exist_ok=True, parents=True)
    model_folder = run_folder / 'models/'
    # if a model has already been trained, just load it to save time
    if not model_folder.exists():
        model_folder.mkdir(exist_ok=True)
        log_dir = run_folder / 'logs/'
        vec_env = make_vec_env(environment, n_envs=6,
                               env_kwargs={'profile': training_profile,
                                           'episode_length': 200,
                                           'run_path': run_folder,
                                           'train_mode': True})
        vec_env = VecMonitor(vec_env,
                            filename=str(log_dir / 'vec'))
        model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1,
                        tensorboard_log=str(log_dir),
                        device='cpu')
        eval_env = envs.HolosSingle(profile=testing_profile,
                                    episode_length=200,
                                    train_mode=False)
        eval_env = Monitor(eval_env, filename=str(log_dir / 'eval'))
        eval_callback = EvalCallback(eval_env=eval_env,
                                     best_model_save_path=str(model_folder),
                                     log_path=str(log_dir),
                                     deterministic=True,
                                     eval_freq=8000)
        model.learn(total_timesteps=2_000_000, callback=eval_callback, progress_bar=True)

    


if __name__ == '__main__':
    main()