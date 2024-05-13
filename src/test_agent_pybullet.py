from pybullet_env import CustomHumanoidDeepBulletEnv

import gym
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

env = CustomHumanoidDeepBulletEnv(renders=True, 
                                  arg_file='run_humanoid3d_dance_b_args.txt', 
                                  custom_cam_dist=2.2, 
                                  custom_cam_pitch=0, 
                                  custom_cam_yaw=90, 
                                #   video_URL=TARGET_VIDEO_URL,
                                #   dataset_pkl_path='bollywood_dance_test.pkl',
                                  dataset_pkl_path='fortnite_floss.pkl',
                                  filename='fortnite_floss',)
                                  # filename = 'bollywood_dance_test')

# model = PPO("MlpPolicy", env, verbose=1, device='cpu')
# model.learn(total_timesteps=100000, progress_bar=True)
# model.save(f"{alg_str}_humanoid_deep_bullet")

# Enjoy trained agent

frames = []
obs = env.reset()
for i in range(60):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    try:
        obs_image = env.render(mode='rgb_array').astype('uint8')
        frames.append(obs_image)
        print(obs_image.shape)
        print('im showing')
    except:
       done = True
    
    #plt.imsave('./imagetest' + str(i) + '.png', obs_image)
    
    print('im showed')
    
    if done:
      obs = env.reset()

# given the list of frames, save as a gif
import imageio
imageio.mimsave('./imagetest.gif', frames)

print('end!')
env.close()