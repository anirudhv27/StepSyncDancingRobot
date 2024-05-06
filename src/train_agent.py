'''
Code to train the agent using the PPO reinforcement learning algorithm. Uses the custom environment and custom reward function to supervise the agent training.
'''

from env import CustomHumanoidDeepBulletEnv

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create and wrap the environment
env_name = "CartPole-v1"
env = make_vec_env(env_name, n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

model.save("ppo_cartpole")
loaded_model = PPO.load("ppo_cartpole", env=env)

obs = env.reset()
for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()