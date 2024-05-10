import gym
from collections import UserDict
import gym.envs.registration

from utils.pose_utils import compute_velocities_between_poses, extract_landmarks_from_frame

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
                 video_URL=None, dataset_pkl_path=None, batch_size=32, learning_rate=0.003, gamma=0.99, gae_lambda=0.95):
        
        self._numSteps = 0
        
        super().__init__(renders=renders, arg_file=arg_file, test_mode=test_mode,
                         time_step=time_step, rescale_actions=rescale_actions, 
                         rescale_observations=rescale_observations)
        
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda

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
                
        filename = 'bollywood_dance_test'
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
    
    '''
    def reset(self):
        print ("Num steps reset")
        print (self._numSteps)
        if self._numSteps is None:
            self._numSteps = 0
        if self._numSteps > 0:
            avg_reward = self.reward_sum / self._numSteps
            print (self.reward_sum, self._numSteps, avg_reward)
            name = "tuning/rewards/"
            name += str(self.batch_size)
            name += "_" + str(self.learning_rate)
            name += "_" + str(self.gamma)
            name += "_" + str(self.gae_lambda)
            name += ".npy"
            try:
                rewards = np.load(name)
            except FileNotFoundError:
                rewards = np.array([])

            rewards = np.append(rewards, avg_reward)
            np.save(name, rewards)

            name_timesteps = "tuning/timesteps/"
            name_timesteps += str(self.batch_size)
            name_timesteps += "_" + str(self.learning_rate)
            name_timesteps += "_" + str(self.gamma)
            name_timesteps += "_" + str(self.gae_lambda)
            name_timesteps += ".npy"
            try:
                timesteps = np.load(name_timesteps)
            except FileNotFoundError:
                timesteps = np.array([])
            timesteps = np.append(timesteps, self._numSteps)
            np.save(name_timesteps, timesteps)

            plot_rewards(self.batch_size, self.learning_rate, self.gamma, self.gae_lambda)
            self.landmark_history = []

            self.reward_sum = 0

        super().reset()
        '''
    
    def step(self, action):
        agent_id = self.agent_id
        done = False

        if self._rescale_actions:
            # Rescale the action
            mean = -self._action_offset
            std = 1./self._action_scale
            action = action * std + mean
        
        # Compute the current pose landmark of the agent
        # Store a history of previous landmarks
        obs_image = self.render(mode='rgb_array').astype('uint8')

        curr_landmarks = extract_landmarks_from_frame(obs_image, self.pose)
        if curr_landmarks is None:         # if not valid, reset the environment
            print('want to reset')
            done = True
            if self._numSteps > 0:
                avg_reward = self.reward_sum / self._numSteps
                #print (self.reward_sum, self._numSteps, avg_reward)
                name = "tuning/rewards/ppo"
                name += str(self.batch_size)
                name += "_" + str(self.learning_rate)
                name += "_" + str(self.gamma)
                name += "_" + str(self.gae_lambda)
                name += ".npy"
                try:
                    rewards = np.load(name)
                except FileNotFoundError:
                    rewards = np.array([])

                rewards = np.append(rewards, avg_reward)
                np.save(name, rewards)

                name_timesteps = "tuning/timesteps/ppo"
                name_timesteps += str(self.batch_size)
                name_timesteps += "_" + str(self.learning_rate)
                name_timesteps += "_" + str(self.gamma)
                name_timesteps += "_" + str(self.gae_lambda)
                name_timesteps += ".npy"
                try:
                    timesteps = np.load(name_timesteps)
                except FileNotFoundError:
                    timesteps = np.array([])
                timesteps = np.append(timesteps, self._numSteps)
                np.save(name_timesteps, timesteps)

                plot_rewards(self.batch_size, self.learning_rate, self.gamma, self.gae_lambda)
                self.reward_sum = 0
                self.landmark_history = []

        else:
            if len(self.landmark_history) >= FRAME_DIFF:
                prev_landmarks = self.landmark_history[-FRAME_DIFF]
                velocity_landmarks = compute_velocities_between_poses(prev_landmarks, curr_landmarks, FRAME_DIFF, FRAMES_PER_SECOND)
                if velocity_landmarks is not None:
                    curr_landmarks['velocity'] = velocity_landmarks['velocity']
                    curr_landmarks['angle_velocity'] = velocity_landmarks['angle_velocity']
        
            self.landmark_history.append(curr_landmarks)

        # Record reward
        reward = self.calc_reward(agent_id, curr_landmarks)
        self.reward_sum += reward

        # Apply control action
        self._internal_env.set_action(agent_id, action)

        start_time = self._internal_env.t

        # step sim
        for i in range(self._num_env_steps):
            self._internal_env.update(self._time_step)

        elapsed_time = self._internal_env.t - start_time

        self._numSteps += 1

        # Record state
        self.state = self._internal_env.record_state(agent_id)
        
        if self._rescale_observations:
            state = np.array(self.state)
            mean = -self._state_offset
            std = 1./self._state_scale 
            state = (state - mean) / (std + 1e-8)

        # Record done
        # done = self._internal_env.is_episode_end()
        done = done or self._internal_env.is_episode_end()
        #print('done1', done)
        info = {}
        print (reward, self._numSteps, self.reward_sum)

        if self._internal_env.is_episode_end():
            print ("Reset", self._numSteps)
            if self._numSteps > 0:
                #print ("logged")
                avg_reward = self.reward_sum / (self._numSteps)
                #print (self.reward_sum, self._numSteps, avg_reward)
                name = "tuning/rewards/ppo"
                name += str(self.batch_size)
                name += "_" + str(self.learning_rate)
                name += "_" + str(self.gamma)
                name += "_" + str(self.gae_lambda)
                name += ".npy"
                try:
                    rewards = np.load(name)
                except FileNotFoundError:
                    rewards = np.array([])

                rewards = np.append(rewards, avg_reward)
                np.save(name, rewards)

                name_timesteps = "tuning/timesteps/ppo"
                name_timesteps += str(self.batch_size)
                name_timesteps += "_" + str(self.learning_rate)
                name_timesteps += "_" + str(self.gamma)
                name_timesteps += "_" + str(self.gae_lambda)
                name_timesteps += ".npy"
                try:
                    timesteps = np.load(name_timesteps)
                except FileNotFoundError:
                    timesteps = np.array([])
                timesteps = np.append(timesteps, self._numSteps)
                np.save(name_timesteps, timesteps)

                plot_rewards(self.batch_size, self.learning_rate, self.gamma, self.gae_lambda)
                self.reward_sum = 0
                self.landmark_history = []

        return state, reward, done, info
        
    
    def quat_diff_sum(self, quat1, quat2):
        quat_diff = 0
        for i in range (quat1.shape[0]):
            # Convert the quaternions to rotation objects
            r1 = R.from_quat(quat1[i])
            r2 = R.from_quat(quat2[i])
            # Calculate the difference between the quaternions
            diff = (r1.inv() * r2).as_quat()
            
            # Calculate the magnitude of the difference
            magnitude = np.linalg.norm(diff, axis=0)

            quat_diff += magnitude
        return quat_diff
    
    def calc_reward(self, agent_id, curr_landmarks):
        # get pose of the current position
        if curr_landmarks is None: # we are about to reset the environment: what reward to give?
            return 0
        
        # get target pose
        target_pose = self.target_poses[self._numSteps]

        agent_pos = np.array(list(curr_landmarks['pos'].values()))
        target_pos = np.array(list(target_pose['pos'].values()))
        
        agent_angle = np.array(list(curr_landmarks['angle'].values()))
        target_angle = np.array(list(target_pose['angle'].values()))
        
        try:
            agent_velocity = np.array(list(curr_landmarks['velocity'].values()))
            target_velocity = np.array(list(target_pose['velocity'].values()))
        except:
            agent_velocity = np.zeros(3)
            target_velocity = np.zeros(3)
        
        try:
            agent_angle_velocity = np.array(list(curr_landmarks['angle_velocity'].values()))
            target_angle_velocity = np.array(list(target_pose['angle_velocity'].values()))
        except:
            agent_angle_velocity = np.zeros(3)
            target_angle_velocity = np.zeros(3)
        
        agent_center_of_mass = np.array(list(curr_landmarks['center_of_mass']))
        target_center_of_mass = np.array(list(target_pose['center_of_mass']))

        # calculate reward
        reward = 0
        angle_quat_diff = np.exp(-2 * self.quat_diff_sum(agent_angle, target_angle))
        weight_angle = 0.65
        reward += weight_angle * angle_quat_diff

        pos_diff = np.linalg.norm(agent_pos - target_pos)
        pos_diff = np.exp(-40 * pos_diff)
        weight_pos = 0.15
        reward += weight_pos * pos_diff

        velocity_diff = np.linalg.norm(agent_angle_velocity - target_angle_velocity)
        velocity_diff = np.exp(-0.1 * velocity_diff)
        weight_velocity = 0.1
        reward += weight_velocity * velocity_diff

        center_of_mass_diff = np.linalg.norm(agent_center_of_mass - target_center_of_mass)
        center_of_mass_diff = np.exp(-10 * center_of_mass_diff)
        weight_center_of_mass = 0.1
        reward += weight_center_of_mass * center_of_mass_diff

        #print(reward)
        return reward