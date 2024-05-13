from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as HumanoidMujocoEnv

import gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# Load the trained model
model = PPO.load("./ppo_humanoid_deep_bullet.zip")
env = HumanoidMujocoEnv()
env.render_mode = 'rgb_array'

# Enjoy trained agent
frames = []
obs, _ = env.reset()
for i in range(150):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    obs_image = env.render().astype('uint8')
    frames.append(obs_image)
    
    if done:
      obs, _ = env.reset()

# given the list of frames, save as a gif
import imageio
imageio.mimsave('./working_agent.gif', frames)

print('end!')
env.close()