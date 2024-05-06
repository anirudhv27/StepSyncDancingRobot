import gym
import pybullet_envs
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv
from utils.dataset_gen import gen_dataset_from_url
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle

import numpy as np

class CustomHumanoidDeepBulletEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        
    def __init__(self, renders=False, arg_file='', test_mode=False,
                 time_step=1./240, rescale_actions=True, rescale_observations=True,
                 custom_cam_dist=4, custom_cam_pitch=0.1, custom_cam_yaw=45,
                 video_URL=None, dataset_pkl_path=None):
        
        super().__init__(renders=renders, arg_file=arg_file, test_mode=test_mode,
                         time_step=time_step, rescale_actions=rescale_actions, 
                         rescale_observations=rescale_observations)
        
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
            self.target_poses = gen_dataset_from_url(video_URL, filename)
            pickle.dump(self.target_poses, open(dataset_pkl_path, 'wb'))
        else:
            self.target_poses = pickle.load(open(dataset_pkl_path, 'rb'))
        
        print(len(self.target_poses))

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

        if self._rescale_actions:
            # Rescale the action
            mean = -self._action_offset
            std = 1./self._action_scale
            action = action * std + mean

        # Record reward
        reward = self.calc_reward(agent_id)

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
        done = self._internal_env.is_episode_end()
        
        info = {}
        return state, reward, done, info
    
    def calc_reward(self, agent_id):
        # raise NotImplementedError
        return self._internal_env.calc_reward(agent_id)