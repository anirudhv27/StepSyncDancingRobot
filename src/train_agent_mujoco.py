'''
Code to train the agent using the PPO reinforcement learning algorithm. Uses the custom environment and custom reward function to supervise the agent training.
'''

from mujoco_env import CustomMujocoEnv

import gym
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# TARGET_VIDEO_URL = 'https://www.youtube.com/watch?v=9TWj9I3CKzg'
TARGET_VIDEO_URL = 'https://www.youtube.com/watch?v=PRdxgTgHAqA'

str_to_alg = {'a2c': A2C, 'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3}

for alg_str in str_to_alg.keys():
    env_kwargs = {
        # 'video_URL': TARGET_VIDEO_URL,
        'dataset_pkl_path': 'fortnite_floss.pkl',
        'filename': 'fortnite_floss',
        'render_mode': 'rgb_array',
        'alg_name': alg_str
    }
    
    env = make_vec_env(CustomMujocoEnv, n_envs=1, env_kwargs=env_kwargs)
    
    alg = str_to_alg[alg_str]
    model = alg("MlpPolicy", env, verbose=1, device='cpu', learning_rate=3e-4)
        
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save(f"{alg_str}_humanoid_deep_bullet")
