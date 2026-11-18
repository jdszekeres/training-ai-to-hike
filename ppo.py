"""
PPO training with Stable-Baselines3 (the previous pure-JAX version is in
legacy/ppo_jax.py).

Environments run in parallel worker processes (SubprocVecEnv). With
preview=True, worker 0 opens a MuJoCo viewer so you can watch that robot
train live with the current policy. Closing the viewer window does not
stop training.
"""
import os
import pickle

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import environment


def make_env(model, goal_position, rank, episode_length=1000, render=False, seed=0, **env_kwargs):
  """Returns a thunk that builds one monitored, time-limited TerrainEnv."""
  def _init():
    env = environment.TerrainEnv(model, goal_position=goal_position, preview=render,
                                 **env_kwargs)
    env = TimeLimit(env, max_episode_steps=episode_length)
    env = Monitor(env)  # episode return/length for SB3's rollout/ep_rew_mean logs
    env.reset(seed=seed + rank)
    return env
  return _init


def make_vec_env(model, goal_position, n_envs, episode_length=1000, preview=False, seed=0,
                 **env_kwargs):
  thunks = [make_env(model, goal_position, i, episode_length, render=preview and i == 0,
                     seed=seed, **env_kwargs) for i in range(n_envs)]
  return SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks)


def train(model, goal_position, total_timesteps=5_000_000, n_envs=None, episode_length=1000,
          preview=False, save_path=os.path.join("checkpoints", "ppo_hike"), checkpoint_every=500_000,
          n_steps=256, batch_size=1024, n_epochs=4, learning_rate=3e-4, ent_coef=0.01,
          gamma=0.99, gae_lambda=0.95, clip_range=0.2, max_grad_norm=1.0,
          hidden_sizes=(256, 256, 128), seed=0, **env_kwargs):
  """
  Trains PPO on TerrainEnv and returns (ppo_model, vec_normalize).

  Each update collects n_envs * n_steps transitions, then runs n_epochs passes
  over them in minibatches of batch_size. Observations and rewards are
  normalized by VecNormalize; its statistics are saved next to the model as
  <save_path>_vecnormalize.pkl and are needed to run the policy later.
  """
  if n_envs is None:
    n_envs = max(1, (os.cpu_count() or 2) - 1)
  venv = make_vec_env(model, goal_position, n_envs, episode_length, preview, seed, **env_kwargs)
  venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)

  agent = PPO(
      "MlpPolicy", venv,
      n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
      learning_rate=learning_rate, ent_coef=ent_coef, vf_coef=0.5,
      gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, max_grad_norm=max_grad_norm,
      policy_kwargs=dict(net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes)),
                         activation_fn=torch.nn.SiLU, log_std_init=-0.5),
      device="gpu",  # small MLPs train faster on CPU than shuttling batches to a GPU
      seed=seed, verbose=1,
  )

  os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
  checkpoints = CheckpointCallback(
      save_freq=max(1, checkpoint_every // n_envs),  # counted in vec-env steps
      save_path=os.path.dirname(save_path) or ".", name_prefix=os.path.basename(save_path),
      save_vecnormalize=True)
  try:
    agent.learn(total_timesteps=total_timesteps, callback=checkpoints, progress_bar=False)
  except KeyboardInterrupt:
    print("Training interrupted; saving the current policy.")
  agent.save(save_path)
  venv.save(save_path + "_vecnormalize.pkl")
  venv.close()
  return agent, venv


def load_policy(save_path=os.path.join("checkpoints", "ppo_hike")):
  """
  Loads a trained agent and returns policy(obs) -> action in [-1, 1]
  (deterministic, with the training-time observation normalization applied).
  """
  agent = PPO.load(save_path, device="cpu")
  vec_norm = None
  if os.path.exists(save_path + "_vecnormalize.pkl"):
    # Only the running statistics are needed, not a venv, so unpickle directly.
    with open(save_path + "_vecnormalize.pkl", "rb") as f:
      vec_norm = pickle.load(f)

  def policy(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if vec_norm is not None:
      obs = vec_norm.normalize_obs(obs)
    action, _ = agent.predict(obs, deterministic=True)
    return action
  return policy
