import stable_baselines3 as sb3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import envs
import microutils
from scipy.interpolate import interp1d
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor


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
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'PID train - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')

    # run the same model on the testing profile
    env = envs.HolosSingle(profile=testing_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.pid_loop(env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'PID test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')

    ####################
    # Single Action RL #
    ####################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'single_action_rl'
    run_folder.mkdir(exist_ok=True, parents=True)
    model_folder = run_folder / 'models/'
    # model_folder.rmdir()
    # if a model has already been trained, just load it to save time
    if not model_folder.exists():
        model_folder.mkdir(exist_ok=True)
        log_dir = run_folder / 'logs/'
        vec_env = make_vec_env(envs.HolosSingle, n_envs=6,
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
                                    train_mode=True)
        eval_env = Monitor(eval_env, filename=str(log_dir / 'eval'))
        eval_callback = EvalCallback(eval_env=eval_env,
                                     best_model_save_path=str(model_folder),
                                     log_path=str(log_dir),
                                     deterministic=True,
                                     eval_freq=2000)
        model.learn(total_timesteps=2_000_000, callback=eval_callback, progress_bar=True)

    # load and run the saved model
    model = sb3.PPO.load(model_folder / 'best_model.zip', device='cpu')
    env = envs.HolosSingle(profile=training_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Single Action RL train - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')

    # run the same model on the testing profile
    env = envs.HolosSingle(profile=testing_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Single Action RL test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')


    ################################
    # Multi Action RL (asymmetric) #
    ################################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'multi_action_rl'
    run_folder.mkdir(exist_ok=True, parents=True)
    model_folder = run_folder / 'models/'
    # model_folder.rmdir()
    # if a model has already been trained, just load it to save time
    if not model_folder.exists():
        model_folder.mkdir(exist_ok=True)
        log_dir = run_folder / 'logs/'
        vec_env = make_vec_env(envs.HolosMulti, n_envs=6,
                               env_kwargs={'profile': training_profile,
                                           'episode_length': 200,
                                           'run_path': run_folder,
                                           'train_mode': True})
        vec_env = VecMonitor(vec_env,
                            filename=str(log_dir / 'vec'))
        model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1,
                        tensorboard_log=str(log_dir),
                        device='cpu')
        eval_env = envs.HolosMulti(profile=testing_profile,
                                    episode_length=200,
                                    train_mode=True)
        eval_env = Monitor(eval_env, filename=str(log_dir / 'eval'))
        eval_callback = EvalCallback(eval_env=eval_env,
                                     best_model_save_path=str(model_folder),
                                     log_path=str(log_dir),
                                     deterministic=True,
                                     eval_freq=2000)
        model.learn(total_timesteps=2_000_000, callback=eval_callback, progress_bar=True)

    # load and run the saved model
    model = sb3.PPO.load(model_folder / 'best_model.zip', device='cpu')
    env = envs.HolosMulti(profile=training_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Multi Action RL asymmetric train - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')

    # run the same model on the testing profile
    env = envs.HolosMulti(profile=testing_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Multi Action RL asymmetric test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')


    ################################
    # Multi Action RL (symmetric) #
    ################################
    # create a run folder
    run_folder = Path.cwd() / 'runs' / 'multi_action_rl_symmetric'
    run_folder.mkdir(exist_ok=True, parents=True)
    model_folder = run_folder / 'models/'
    # model_folder.rmdir()
    # if a model has already been trained, just load it to save time
    if not model_folder.exists():
        model_folder.mkdir(exist_ok=True)
        log_dir = run_folder / 'logs/'
        vec_env = make_vec_env(envs.HolosMulti, n_envs=6,
                               env_kwargs={'profile': training_profile,
                                           'episode_length': 200,
                                           'run_path': run_folder,
                                           'train_mode': True,
                                           'symmetry_reward': True})
        vec_env = VecMonitor(vec_env,
                            filename=str(log_dir / 'vec'))
        model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1,
                        tensorboard_log=str(log_dir),
                        device='cpu')
        eval_env = envs.HolosMulti(profile=testing_profile,
                                    episode_length=200,
                                    train_mode=True,
                                    symmetry_reward=True)
        eval_env = Monitor(eval_env, filename=str(log_dir / 'eval'))
        eval_callback = EvalCallback(eval_env=eval_env,
                                     best_model_save_path=str(model_folder),
                                     log_path=str(log_dir),
                                     deterministic=True,
                                     eval_freq=2000)
        model.learn(total_timesteps=2_000_000, callback=eval_callback, progress_bar=True)

    # load and run the saved model
    model = sb3.PPO.load(model_folder / 'best_model.zip', device='cpu')
    env = envs.HolosMulti(profile=training_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Single Action RL train - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')

    # run the same model on the testing profile
    env = envs.HolosMulti(profile=testing_profile, episode_length=200, run_path=run_folder, train_mode=False)
    microutils.rl_control_loop(model, env)
    history_path = microutils.find_latest_file(run_folder, pattern='run_history*.csv')
    history = microutils.load_history(history_path)
    # microutils.plot_history(history)
    mae, iae, control_effort = microutils.calc_metrics(history)
    print(f'Single Action RL test - MAE: {mae}, IAE: {iae}, Control Effort: {control_effort}')


if __name__ == '__main__':
    main()