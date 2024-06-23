import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
import time

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
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))

random_seeds = [12]

# Specify a general directory
tensorboard_dir = "PPO_tensorboard_logs/PPO_12/"
os.makedirs(tensorboard_dir, exist_ok=True)

# # Saving tensorboard logs for plotting later
# log_dir = f"./walker_PPO_tensorboard/"
# full_log_dir = os.path.join(tensorboard_dir, log_dir)
# os.makedirs(full_log_dir, exist_ok=True)
# # Create the environment

for seed in random_seeds:
  log_dir = f"./walker_PPO_seed_{seed}_tensorboard/"
  full_log_dir = os.path.join(tensorboard_dir, log_dir)
  os.makedirs(full_log_dir, exist_ok=True)
  # Create the environment
  env = make_vec_env('BipedalWalker-v3', n_envs = 4, seed = seed)
  env = VecNormalize(env, norm_obs=True, norm_reward = True)


  callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=350, verbose=1)
  eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
  time_and_step_callback = TimeAndStepTrackingCallback()

  # Use a list of callbacks
  callback = [eval_callback, time_and_step_callback]
  # Define the model
  model = PPO("MlpPolicy",
                batch_size = 32,
                n_steps = 512,
                gamma = 0.99,
                gae_lambda = 0.9,
                ent_coef =  6.243275791612454e-06,
                clip_range = 0.2,
                n_epochs = 5,
                max_grad_norm = 0.6,
                env = env,
                verbose =0,
                vf_coef = 0.38040715408689363,
                policy_kwargs=policy_kwargs,
                target_kl = 0.1200865249782578,
                learning_rate = 0.00018451057856134118,
                tensorboard_log=full_log_dir)
  model.learn(total_timesteps = 10000000, callback=callback)
  model.save(f'PPO_walker_seed_{seed}_24h')
  env.save(f'PPO_walker_seed_{seed}_24h.pkl')