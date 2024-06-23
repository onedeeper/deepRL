import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TRPO
import torch as th
import os

class ChangeMassWrapper(gym.Wrapper): 
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass


# define torso masses
masses = [3,6,9]

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))

for m in masses:
    #log_dir = f"./hopper_{m}_tr_1_kg_tensorboard/"
    #full_log_dir = os.path.join(tensorboard_dir, log_dir)
    #os.makedirs(full_log_dir, exist_ok=True)
    # Create the environment
    env = make_vec_env('Hopper-v4', n_envs = 4, seed = 1, wrapper_class = ChangeMassWrapper, wrapper_kwargs = {'torso_mass':m})
    env = VecNormalize(env, norm_obs=True, norm_reward = True)
    # Define the model
    model = TRPO("MlpPolicy",
                 n_steps = 2048,
                 gamma = 0.99,
                 gae_lambda = 0.95,
                 env = env,
                 verbose =0,
                 policy_kwargs=policy_kwargs,
                 target_kl = 0.13,
                 learning_rate = 0.0002,
                 cg_max_steps =10,
                 cg_damping = 0.1)
    model.learn(total_timesteps = 1000000)
    model.save(f'model_{m}_kg')
    env.save(f'env_{m}_kg.pkl')