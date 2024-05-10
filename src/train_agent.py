'''
Code to train the agent using the PPO reinforcement learning algorithm. Uses the custom environment and custom reward function to supervise the agent training.
'''

from env import CustomHumanoidDeepBulletEnv

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

TARGET_VIDEO_URL = 'https://www.youtube.com/watch?v=9TWj9I3CKzg'

for batch_size in [32, 64]:
    for learning_rate in [0.0001, 0.0005]:
        for gamma in [0.9, 0.95, 0.99]:
            for gae_lambda in [0.8, 0.9]:
                env = CustomHumanoidDeepBulletEnv(renders=False, 
                                  arg_file='run_humanoid3d_dance_b_args.txt', 
                                  custom_cam_dist=2.2, 
                                  custom_cam_pitch=0, 
                                  custom_cam_yaw=90, 
                                  #video_URL=TARGET_VIDEO_URL,
                                  dataset_pkl_path='bollywood_dance_test.pkl',
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  gamma=gamma,
                                  gae_lambda=gae_lambda)
                model = PPO("MlpPolicy", env, verbose=1, n_steps=100000, batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda)
                model.learn(total_timesteps=100000, progress_bar=True)
                model.save("ppo_humanoid_deep_bullet" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda))

'''
env = CustomHumanoidDeepBulletEnv(renders=True, 
                                  arg_file='run_humanoid3d_dance_b_args.txt', 
                                  custom_cam_dist=2.2, 
                                  custom_cam_pitch=0, 
                                  custom_cam_yaw=90, 
                                  #video_URL=TARGET_VIDEO_URL,
                                  dataset_pkl_path='bollywood_dance_test.pkl')

print('initialized env')

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, n_steps = 10)

# Train the agent
model.learn(total_timesteps=10000, progress_bar=True)

# Save the model
model.save("ppo_humanoid_deep_bullet")

# Load the model
model = PPO.load("ppo_humanoid_deep_bullet")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()'''