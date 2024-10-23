import imageio  
from pathlib import Path
import argparse
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from micro_gym import MicroEnv, create_gif
from controllers import PIDController


# TODO Run n_steps from 4 to 4000
def train_model_loop(run_name: str,
                     num_timesteps: int = 1000000,
                     num_envs: int = 6,
                     n_steps: int = 2048,
                     pretrained_model_path: Path = None,
                     pretrained_timesteps: int = 0,
                     ):
    # tensorboard_dir = f'./runs/tb_logs/'
    tensorboard_dir = f'./runs/{run_name}/logs/'
    vec_env = make_vec_env(MicroEnv, n_envs=num_envs,
                           env_kwargs={'render_mode': None})
    vec_env = VecMonitor(vec_env,
                         filename=f'./runs/{run_name}/logs/vec')
    if pretrained_model_path is not None:
        model = sb3.PPO.load(pretrained_model_path, env=vec_env,
                             tensorboard_log=tensorboard_dir)
        model.num_timesteps = pretrained_timesteps
    else:
        model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1,
                        n_steps=n_steps,
                        tensorboard_log=tensorboard_dir)

    eval_env = MicroEnv()
    eval_env = Monitor(eval_env, filename=f'./runs/{run_name}/logs/eval')
    eval_callback = EvalCallback(eval_env=eval_env,
                                     best_model_save_path=f'./runs/{run_name}',
                                     log_path=f'./runs/{run_name}/logs/',
                                     deterministic=True,
                                     eval_freq=1000)

    model.learn(total_timesteps=num_timesteps, callback=eval_callback,
                reset_num_timesteps=False)
    model.save(f'./runs/{run_name}/{model.num_timesteps}.zip')


def load_model_loop(run_name: str, model_path: Path):
    env = MicroEnv(run_name=run_name)
    ppo_controller = sb3.PPO.load(model_path)
    
    obs, _ = env.reset()
    rewards = []
    
    done = False
    while not done:
        env.render()
        gym_action, _states = ppo_controller.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(gym_action)
        rewards.append(reward)
        if terminated or truncated:
            done = True


def pid_loop(run_name: str, run_folder: Path):
    # create a microreactor simulator and a PID controller
    env = MicroEnv(run_name=run_name)
    pid = PIDController()

    for _ in range(1):
        env.reset()
        done = False
        action = 0
        while not done:
            env.render()
            gym_action = env.convert_action_to_gym(action)
            obs, _, terminated, truncated, _ = env.step(gym_action)
            action = pid.update(env.time, obs["power"], env.desired_profile(env.time))
            if terminated or truncated:
                done = True

def main(args):
    run_name = args.run_name
    run_folder = Path.cwd() / 'runs' / run_name
    run_folder.mkdir(exist_ok=True, parents=True)

    match args.run_type:
        case 'train':
            saved_models = list(run_folder.glob('*[0-9].zip'))
            latest_model = None
            pretrained_timesteps = 0
            if len(saved_models) > 0:
                latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[-1]
                pretrained_timesteps = int(latest_model.stem)
            train_model_loop(run_name, num_envs=args.num_envs,
                             num_timesteps=args.num_timesteps,
                             n_steps=args.nsteps,
                             pretrained_model_path=latest_model,
                             pretrained_timesteps=pretrained_timesteps)
            best_model = list(run_folder.glob('best_model.zip'))[0]
            saved_models = list(run_folder.glob('*[0-9].zip'))
            if len(saved_models) > 0:
                latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[-1]
            load_model_loop(run_name, latest_model)
        case 'load':
            best_model = list(run_folder.glob('best_model.zip'))[0]
            saved_models = list(run_folder.glob('*[0-9].zip'))
            if len(saved_models) > 0:
                latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[-1]
            load_model_loop(run_name, best_model)
        case 'pid':
            pid_loop(run_name, run_folder)
        case _:
            print('Invalid run type')
            return

    create_gif(run_name, png_folder=run_folder)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a gif from pngs')
    parser.add_argument('run_name', type=str, help='Name of the run')
    parser.add_argument('--run_type', type=str, default='train',
                        help='Must be train, load, or pid')
    parser.add_argument('--num_timesteps', type=int, default=500000,)
    parser.add_argument('--num_envs', type=int, default=6,
                        help='Number of parallel environments during training')
    parser.add_argument('--nsteps', type=int, default=2048,
                        help='PPO hyperparameter')
    
    args = parser.parse_args()
    main(args)
