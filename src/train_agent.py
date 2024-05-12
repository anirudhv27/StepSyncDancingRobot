'''
Code to train the agent using the PPO reinforcement learning algorithm. Uses the custom environment and custom reward function to supervise the agent training.
'''

from pybullet_env import CustomHumanoidDeepBulletEnv

import gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# TARGET_VIDEO_URL = 'https://www.youtube.com/watch?v=9TWj9I3CKzg'
TARGET_VIDEO_URL = 'https://www.youtube.com/watch?v=PRdxgTgHAqA'

str_to_alg = {'a2c': A2C, 'ddpg': DDPG}

for alg_str in ['a2c', 'ddpg']:
    # env = CustomHumanoidDeepBulletEnv(renders=False, 
    #     arg_file='run_humanoid3d_dance_b_args.txt', 
    #     custom_cam_dist=2.2, 
    #     custom_cam_pitch=0, 
    #     custom_cam_yaw=90, 
    #     # video_URL=TARGET_VIDEO_URL,
    #     #   dataset_pkl_path='bollywood_dance_test.pkl',
    #     dataset_pkl_path='fortnite_floss.pkl',
    #     filename='fortnite_floss',
    #     # filename = 'bollywood_dance_test'
    #     alg_name=alg_str)
    
    env_kwargs = {
        'arg_file': 'run_humanoid3d_dance_b_args.txt', 
        'custom_cam_dist': 2.2, 
        'custom_cam_pitch': 0, 
        'custom_cam_yaw': 90, 
        # 'video_URL': TARGET_VIDEO_URL,
        'dataset_pkl_path': 'fortnite_floss.pkl',
        'filename': 'fortnite_floss',
        'alg_name': alg_str
    }
    
    env = make_vec_env(CustomHumanoidDeepBulletEnv, n_envs=4, env_kwargs=env_kwargs)
    
    alg = str_to_alg[alg_str]
    model = alg("MlpPolicy", env, verbose=1, n_steps=1, device='mps')
        
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save(f"{alg_str}_humanoid_deep_bullet")
