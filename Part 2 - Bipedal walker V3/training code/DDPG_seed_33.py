import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import torch
import os
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
import time
from datetime import datetime

class TimeAndStepTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TimeAndStepTrackingCallback, self).__init__(verbose)
        self.start_time = None
        self.steps = 0

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        self.steps += 1
        return True

    def _on_training_end(self):
        elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        print(f"Total environment steps: {self.steps}")

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64], qf=[64,64]))

random_seeds = [33]

# Specify a general directory
tensorboard_dir = "DDPG_tensorboard_logs/"
os.makedirs(tensorboard_dir, exist_ok=True)

for seed in random_seeds:
# Saving tensorboard logs for plotting later
  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  log_dir = f"./walker_DDPG_{seed}_tensorboard_{timestamp}/"
  full_log_dir = os.path.join(tensorboard_dir, log_dir)
  os.makedirs(full_log_dir, exist_ok=True)

  print(f"Seed: {seed}, Full Log Dir: {full_log_dir}")
  # Create the environment
  env = make_vec_env('BipedalWalker-v3', n_envs = 4, seed = seed)
  env = VecNormalize(env, norm_obs=True, norm_reward = True)

  callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
  eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
  time_and_step_callback = TimeAndStepTrackingCallback()
  # Use a list of callbacks
  callback = [eval_callback, time_and_step_callback]
  # Define the model according to optimized values (trained seperately on cluster as described in paper)

  n_actions = env.action_space.shape[0]
  model = DDPG("MlpPolicy",
                gamma = 0.98,
                learning_rate = 2.7076207562598673e-05,
                batch_size = 512,
                buffer_size = 1000000,
                tau = 0.001,
                train_freq = 16,
                action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma = 0.8906602621112266),
                env = env,
                verbose =0,
                policy_kwargs=policy_kwargs,
                tensorboard_log=full_log_dir)
  model.learn(total_timesteps = 10000000, callback=callback)
  model.save(f'DDPG_walker_seed{seed}_24h_{timestamp}')
  env.save(f'DDPG_walker_seed_{seed}_24h_{timestamp}.pkl')