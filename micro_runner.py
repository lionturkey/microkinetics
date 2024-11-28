import imageio  
from pathlib import Path
import shutil
import argparse
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from micro_gym import MicroEnv, pid_loop


def create_gif(run_name: str, png_folder: Path = (Path.cwd() / 'runs')):
    # Build GIF
    png_list = list(png_folder.glob('*.png'))
    num_list = sorted([int(png.stem) for png in png_list])
    png_list = [(png_folder / f'{i}.png') for i in num_list]
    with imageio.get_writer((png_folder / f'{run_name}.gif'), mode='I') as writer:
        for filepath in png_list[::2]:
            image = imageio.imread(filepath)
            writer.append_data(image)
        for filepath in png_list:
            # delete png
            filepath.unlink()


def load_model_loop(model_path: Path, env):
    ppo_controller = sb3.PPO.load(model_path)
    
    obs, _ = env.reset()
    rewards = []
    
    done = False
    while not done:
        gym_action, _states = ppo_controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(gym_action)
        rewards.append(reward)
        if terminated or truncated:
            done = True
    env.render()


def check_zip(run_folder: Path):
    latest_model = None
    best_model = None

    best_models = list(run_folder.glob('best_model.zip'))
    saved_models = list(run_folder.glob('*[0-9].zip'))

    if len(saved_models) > 1:
        latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[-1]
    if len(best_models) == 1:
        best_model = best_models[0]

    return latest_model, best_model


def post_process(run_folder: Path):
    """
    If the best model occured long ago, fake the latest model to be best"""
    saved_models = list(run_folder.glob('*[0-9].zip'))
    latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[-1]
    best_model = list(run_folder.glob('best_model.zip'))[0]

    time_diff = latest_model.stat().st_mtime - best_model.stat().st_mtime
    if time_diff > 100:
        fake_latest = latest_model.parent / '69696969.zip'  # assuming we'll never train this far...
        shutil.copyfile(best_model, fake_latest)


def main(args):
    run_name = args.run_name
    run_folder = Path.cwd() / 'runs' / run_name
    run_folder.mkdir(exist_ok=True, parents=True)
    latest_model, best_model = check_zip(run_folder)
    pretrained_timesteps = int(latest_model.stem) if latest_model else 0
    run_kwargs = {'run_name': run_name,
                  'run_mode': args.run_type,
                  'noise': args.noise,
                  'profile': args.profile,
                  'reward_mode': args.reward,
                  'scale_graphs': True,
                  }
    eval_env = MicroEnv(**run_kwargs)

    if args.run_type == 'train':
        tensorboard_dir = f'./runs/{run_name}/logs/'
        vec_env = make_vec_env(MicroEnv, n_envs=args.num_envs,
                               env_kwargs=run_kwargs)
        vec_env = VecMonitor(vec_env,
                             filename=f'./runs/{run_name}/logs/vec')
        if latest_model:
            model = sb3.PPO.load(latest_model, env=vec_env, verbose=1,
                                 n_steps=args.nsteps,
                                 tensorboard_log=tensorboard_dir)
        else:
            model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1,
                            n_steps=args.nsteps,
                            tensorboard_log=tensorboard_dir)

        model.num_timesteps = pretrained_timesteps
        eval_env = Monitor(eval_env, filename=f'./runs/{run_name}/logs/eval')
        eval_callback = EvalCallback(eval_env=eval_env,
                                        best_model_save_path=f'./runs/{run_name}',
                                        log_path=f'./runs/{run_name}/logs/',
                                        deterministic=True,
                                        eval_freq=4000)

        model.learn(total_timesteps=args.num_timesteps, callback=eval_callback,
                    reset_num_timesteps=False, progress_bar=True)
        model.save(f'./runs/{run_name}/{model.num_timesteps}.zip')
        post_process(run_folder)
        latest_model, best_model = check_zip(run_folder)
    elif args.run_type == 'pid':
        pid_loop(eval_env)
        return  # exit

    # Load best model
    if best_model:
        load_model_loop(best_model, eval_env)
    else:
        print('Error: no best model found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a gif from pngs')
    parser.add_argument('run_name', type=str, help='Name of the run')
    parser.add_argument('--run_type', type=str, default='train',
                        help='Must be train, load, or pid')
    parser.add_argument('--num_timesteps', type=int, default=800000,)
    parser.add_argument('--num_envs', type=int, default=6,
                        help='Number of parallel environments during training')
    parser.add_argument('--nsteps', type=int, default=2048,
                        help='PPO hyperparameter')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='stdev of normal noise to add to observations')
    parser.add_argument('--reward', type=str, default='optimal',
                        help='optimal, frugal, or sleepy')
    parser.add_argument('--profile', type=str, default='train',
                        help='train, test, longtest, power0, xe20, xe20power0')
    
    args = parser.parse_args()
    main(args)
