from pettingzoo.utils.conversions import parallel_to_aec
from micro_gym import MicroEnv
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from pettingzoo import ParallelEnv
from typing import Any, Dict as TypeDict, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import supersuit as ss
from pathlib import Path


class MARLMicroEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "marl_micro_v0"}

    def __init__(self, render_mode=None, num_drums=8, max_masks=3,
                 min_masks=0,
                 run_name=None,
                 profile='train',
                 **kwargs):
        super().__init__()
        self.env = MicroEnv(render_mode=render_mode, run_name=run_name, profile=profile, **kwargs)
        self.run_name = run_name
        self.agents = [f"agent_{i}" for i in range(num_drums)]
        self.possible_agents = self.agents[:]
        self.max_masks = max_masks
        self.min_masks = min_masks
        self.render_mode = render_mode

        # Each agent contributes 1/8th of the total action
        self._action_spaces = {
            agent: Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Each agent gets the same observations
        self._observation_spaces = {
            agent: Dict({
                "last_action": Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "last_power": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "power": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "next_desired_power": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
            for agent in self.agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        obs, _ = self.env.reset(seed=seed)
        self.last_power = obs["power"]
        obs["last_power"] = self.last_power
        
        observations = {agent: obs for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.masks = {agent: 1 for agent in self.agents}
        num_masks = np.random.randint(self.min_masks, self.max_masks + 1)
        disabled_agents = np.random.choice(self.agents, size=num_masks, replace=False)
        for agent in disabled_agents:
            self.masks[agent] = 0
        # self.infos["masks"] = self.masks

        return observations, self.infos

    def step(self, actions):
        # Combine actions from all agents
        total_action = sum(actions[agent]*self.masks[agent] for agent in self.agents) / self.num_agents
        
        # Step the environment with combined action
        obs, reward, terminated, truncated, info = self.env.step(total_action)
        
        # Distribute observations, rewards, and other info to all agents
        observations = {}
        for agent in self.agents:
            observations[agent] = obs.copy()
            observations[agent]["last_power"] = self.last_power
            observations[agent]["last_action"] = actions[agent]
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def train_loop(env, run_folder):
    env.reset()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 6, base_class="stable_baselines3")
    vec_log_folder = run_folder / 'logs/vec'
    env = VecMonitor(env, filename=str(vec_log_folder))
    tensorboard_dir = run_folder / 'logs'
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=str(run_folder),
                                             name_prefix='ppo_marl')
    # eval_env = parallel_to_aec(env)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=run_folder,
    #                              log_path=run_folder / 'logs', eval_freq=50_000)
    # callbacks = CallbackList([checkpoint_callback, eval_callback])

    saved_models = list(run_folder.glob('*_steps.zip'))
    if saved_models:
        latest_model = max(saved_models, key=lambda x: int(str(x).split('_')[-2]))
        model = PPO.load(latest_model, device='cpu', env=env, verbose=1,
                         tensorboard_log=str(tensorboard_dir), print_system_info=True)
        current_timesteps = int(str(latest_model).split('_')[-2])
        model.n_steps = current_timesteps
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=str(tensorboard_dir), device='cpu')

    # model.learn(total_timesteps=20_000_000, callback=checkpoint_callback, progress_bar=True)
    model.learn(total_timesteps=20_000_000, callback=checkpoint_callback,
                progress_bar=True, reset_num_timesteps=False)

    model.save(run_folder / 'ppo_marl')


def eval_loop(env, run_folder, checkpoint_num=40):
    env = parallel_to_aec(env)
    nsteps = checkpoint_num * 480_000
    model = PPO.load(run_folder / f'ppo_marl_{nsteps}_steps.zip', device='cpu')

    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            break
        else:
            action, _ = model.predict(obs, deterministic=True)
            env.step(action)

    env.render()


# Example usage
if __name__ == "__main__":
    run_name = 'marl_timeline2'
    # profile = 'train'
    profile = 'test'
    # run_type = 'train'
    run_type = 'eval'
    render_mode = None
    checkpoint_num = 40
    # render_mode = 'human'
    # run_type = 'eval'

    bad_drums = 4
    if bad_drums >= 0:
        min_masks = bad_drums
        max_masks = bad_drums
    else:
        min_masks = 0
        max_masks = 3
    env = MARLMicroEnv(run_name=run_name, profile=profile, render_mode=render_mode, min_masks=min_masks, max_masks=max_masks)

    run_folder = Path.cwd() / 'runs' / run_name
    run_folder.mkdir(exist_ok=True, parents=True)
    
    if run_type == 'train':
        train_loop(env, run_folder)
        eval_loop(env, run_folder)
    elif run_type == 'eval':
        eval_loop(env, run_folder)
    else:
        # Run for a few steps
        for _ in range(200):
            # Random actions for each agent
            actions = {
                agent: env._action_spaces[agent].sample()
                for agent in env.agents
            }
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break
                observations, _ = env.reset()
        env.render()