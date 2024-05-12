import gym
from collections import UserDict
import gym.envs.registration

from utils.pose_utils import compute_velocities_between_poses, extract_landmarks_from_frame
from utils.reward_utils import calc_reward

registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv
from utils.videoProcessing import generate_dataset_from_url
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R
from plot_rewards import plot_rewards

FRAME_DIFF = 3
FRAMES_PER_SECOND = 30

class CustomHumanoidDeepBulletEnv(HumanoidDeepBulletEnv):
    # metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': FRAMES_PER_SECOND}
    
    def __init__(self, renders=False, arg_file='', test_mode=False,
                 time_step=1./240, rescale_actions=True, rescale_observations=True,
                 custom_cam_dist=4, custom_cam_pitch=0.1, custom_cam_yaw=45,
                 video_URL=None, dataset_pkl_path=None, filename='fortnite_floss', alg_name='ppo'):
        
        self._numSteps = 0
        
        super().__init__(renders=renders, arg_file=arg_file, test_mode=test_mode,
                         time_step=time_step, rescale_actions=rescale_actions, 
                         rescale_observations=rescale_observations)
        
        
        # self.batch_size = batch_size
        # self.learning_rate = learning_rate
        # self.gamma = gamma
        # self.gae_lambda = gae_lambda
        self.alg_name = alg_name

        self._cam_dist = custom_cam_dist
        self._cam_pitch = custom_cam_pitch
        self._cam_yaw = custom_cam_yaw
        
        # Initialize mediapipe estimator
        print('Initializing mediapipe estimator...')
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) # , model_complexity={0,1,2} (fastest to slowest)    
        
        # If dataset_pkl is none, download video from URL
        if dataset_pkl_path is None:
            raise ValueError('dataset_pkl_path cannot be None')
        
        if video_URL is not None:
            print('Downloading video...')
            self.target_poses = generate_dataset_from_url(video_URL, filename)
            pickle.dump(self.target_poses, open(dataset_pkl_path, 'wb'))
        else:
            self.target_poses = pickle.load(open(dataset_pkl_path, 'rb'))
        
        print('dataset length: ', len(self.target_poses))
        self.landmark_history = []
        self.reward_sum = 0

    def render(self, mode='human', close=False):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        human = self._internal_env._humanoid
        base_pos, orn = self._p.getBasePositionAndOrientation(human._sim_model)
        base_pos = np.asarray(base_pos)
        # track the position
        base_pos[1] += 0.1
        rpy = self._p.getEulerFromQuaternion(orn)  # rpy, in radians
        rpy = 180 / np.pi * np.asarray(rpy)  # convert rpy in degrees

        if (not self._p == None):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=1)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                    aspect=float(self._render_width) / self._render_height,
                    nearVal=0.1,
                    farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def step(self, action):
        agent_id = self.agent_id
        done = False

        if self._rescale_actions:
            # Rescale the action
            mean = -self._action_offset
            std = 1./self._action_scale
            action = action * std + mean
        
         # Apply control action
        self._internal_env.set_action(agent_id, action)

        start_time = self._internal_env.t

        # step sim
        for i in range(self._num_env_steps):
            self._internal_env.update(self._time_step)

        elapsed_time = self._internal_env.t - start_time


        # Record state
        self.state = self._internal_env.record_state(agent_id)
        
        if self._rescale_observations:
            state = np.array(self.state)
            mean = -self._state_offset
            std = 1./self._state_scale 
            state = (state - mean) / (std + 1e-8)
        
        self._numSteps += 1
        # Record done if humanoid has fallen
        done = self._internal_env.is_episode_end()
        
        # Compute reward from this new position
        # Compute the current pose landmark of the agent, and add to the 
        # Store a history of previous landmarks
        obs_image = self.render(mode='rgb_array').astype('uint8')        
        curr_landmarks = extract_landmarks_from_frame(obs_image, self.pose)



        if curr_landmarks is None:
            # print('can\'t read landmarks from this state!')
            done = True # set that i want to reset
            
        elif done:
            # print('humanoid has fallen; do not compute reward')
            pass
        else:
            if len(self.landmark_history) >= FRAME_DIFF:
                prev_landmarks = self.landmark_history[-FRAME_DIFF]
                velocity_landmarks = compute_velocities_between_poses(prev_landmarks, curr_landmarks, FRAME_DIFF, FRAMES_PER_SECOND)
                if velocity_landmarks is not None:
                    curr_landmarks['velocity'] = velocity_landmarks['velocity']
                    curr_landmarks['angle_velocity'] = velocity_landmarks['angle_velocity']
        
            self.landmark_history.append(curr_landmarks)
            # Don't reset, instead continue to append to this episode's history
        
        # Compute reward from the landmarks that are read
        reward = calc_reward(self.target_poses, self._numSteps, agent_id, curr_landmarks, done)
        
        self.reward_sum += reward
        if done:
            # About to reset, which means that we need to save the average rewards and timesteps to failure
            ep_avg_reward = self.reward_sum / self._numSteps # only the average reward until failing
            name = f"tuning/rewards/pybullet_{self.alg_name}"
            # name += str(self.batch_size)
            # name += "_" + str(self.learning_rate)
            # name += "_" + str(self.gamma)
            # name += "_" + str(self.gae_lambda)
            name += ".npy"
            try:
                rewards = np.load(name)
            except FileNotFoundError:
                rewards = np.array([])

            rewards = np.append(rewards, ep_avg_reward)
            np.save(name, rewards)

            name_timesteps = f"tuning/timesteps/pybullet_{self.alg_name}"
            # name_timesteps += str(self.batch_size)
            # name_timesteps += "_" + str(self.learning_rate)
            # name_timesteps += "_" + str(self.gamma)
            # name_timesteps += "_" + str(self.gae_lambda)
            name_timesteps += ".npy"
            try:
                timesteps = np.load(name_timesteps)
            except FileNotFoundError:
                timesteps = np.array([])
            timesteps = np.append(timesteps, self._numSteps)
            np.save(name_timesteps, timesteps)

            plot_rewards('pybullet', self.alg_name)
            
            # After saving, we want to reset episode-level statistics
            self.reward_sum = 0
            self.landmark_history = []

        info = {}
        # print(reward, self._numSteps, self.reward_sum)
        # if done: print('Done and logged!')

        return state, reward, done, info