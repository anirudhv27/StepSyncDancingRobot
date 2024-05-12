from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as HumanoidMujocoEnv

import gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

env = HumanoidMujocoEnv()
env.render_mode = 'rgb_array'

# model = PPO("MlpPolicy", env, verbose=1, device='cpu')
# model.learn(total_timesteps=100000, progress_bar=True)
# model.save(f"{alg_str}_humanoid_deep_bullet")

# Enjoy trained agent
obs = env.reset()
for i in range(1):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    obs_image = env.render().astype('uint8')
    print(obs_image.shape)    
    print('im showing')
    
    
    plt.imsave('./imagetest.png', obs_image)
    
    print('im showed')
    
    if done:
      obs = env.reset()

print('end!')
env.close()