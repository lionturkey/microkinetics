import imageio  
from pathlib import Path
import argparse
from pid_lit import MicroEnv, PIDController
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

def create_gif(run_name: str, png_folder: Path = (Path.cwd() / 'runs')):
    # Build GIF
    png_list = [(png_folder / f'{i}.png') for i in range(200)]
    with imageio.get_writer((png_folder / f'{run_name}.gif'), mode='I') as writer:
        for filepath in png_list:
            image = imageio.imread(filepath)
            writer.append_data(image)
            # delete png
            filepath.unlink()


# TODO Add training loop with callback to save (20 per test)
# TODO Run n_steps from 4 to 4000
def train_model_loop(run_name: str, num_timesteps: int = 10000000,
                     num_checkpoints: int = 20, num_envs: int = 6,
                     n_steps: int = 200):
    vec_env = make_vec_env(MicroEnv, n_envs=num_envs,
                           env_kwargs={'render_mode': None})
    model = sb3.PPO('MultiInputPolicy', vec_env, verbose=1, n_steps=n_steps)

    checksteps = round(num_timesteps/num_checkpoints)
    checkpoint_callback = CheckpointCallback(save_freq=checksteps,
                                             save_path=f'./runs/{run_name}',
                                             name_prefix=f'PPO_{run_name}')
    model.learn(total_timesteps=num_timesteps, callback=checkpoint_callback)
    model.save(f'ppo_microreactor_{num_timesteps}')


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
            action = pid.update(env.t, obs["power"], env.profile(env.t))
            if terminated or truncated:
                done = True

def main(args):
    run_name = args.run_name
    run_folder = Path.cwd() / 'runs' / run_name
    run_folder.mkdir(exist_ok=True, parents=True)

    match args.run_type:
        case 'train':
            train_model_loop(run_name, num_envs=args.num_envs,
                             num_timesteps=args.num_timesteps,
                             n_steps=args.nsteps)
            saved_models = list(run_folder.glob('*.zip'))
            latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[0]
            load_model_loop(run_name, latest_model)
        case 'load':
            saved_models = list(run_folder.glob('*.zip'))
            latest_model = sorted(saved_models, key=lambda x: x.stat().st_mtime)[0]
            load_model_loop(run_name, latest_model)
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
    parser.add_argument('--num_timesteps', type=int, default=2500000,)
    parser.add_argument('--num_envs', type=int, default=6,
                        help='Number of parallel environments during training')
    parser.add_argument('--nsteps', type=int, default=4,
                        help='PPO hyperparameter')
    
    args = parser.parse_args()
    main(args)
