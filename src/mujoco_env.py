from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as HumanoidMujocoEnv
import numpy as np

from plot_rewards import plot_rewards
from utils.pose_utils import compute_velocities_between_poses, extract_landmarks_from_frame
from utils.reward_utils import calc_reward
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mediapipe as mp
import pickle
import matplotlib.pyplot as plt

from utils.videoProcessing import generate_dataset_from_url

FRAME_DIFF = 3
FRAMES_PER_SECOND = 30

# Todo: modified by ani!
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -15.0,
    "azimuth": 180
}

class CustomMujocoEnv(HumanoidMujocoEnv):
    def __init__(self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        video_URL=None, dataset_pkl_path=None, filename='fortnite_floss', alg_name='ppo',
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            video_URL, dataset_pkl_path, filename, alg_name,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )
        
        self.alg_name = alg_name
        self._numSteps = 0
        
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

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self._numSteps += 1
        observation = self._get_obs()
        terminated = self.terminated
        
        # Compute reward from current position after action
        agent_id = 0 # placeholder
        obs_image = self.render().astype('uint8')
        plt.imsave('./image.png', obs_image)
        curr_landmarks = extract_landmarks_from_frame(obs_image, self.pose)
        
        if curr_landmarks is None:
            # print('cant read pose from this state')
            terminated = True
        elif terminated:
            # print('humanoid has fallen, do not compute reward!')
            pass
        else:
            if len(self.landmark_history) >= FRAME_DIFF:
                prev_landmarks = self.landmark_history[-FRAME_DIFF]
                velocity_landmarks = compute_velocities_between_poses(prev_landmarks, curr_landmarks, FRAME_DIFF, FRAMES_PER_SECOND)
                if velocity_landmarks is not None:
                    curr_landmarks['velocity'] = velocity_landmarks['velocity']
                    curr_landmarks['angle_velocity'] = velocity_landmarks['angle_velocity']
        
            self.landmark_history.append(curr_landmarks)
        
        reward = calc_reward(self.target_poses, self._numSteps, agent_id, curr_landmarks, terminated)
        reward += (5.0 if not terminated else 0.0)
        self.reward_sum += reward
        
        if terminated:
            # About to reset, which means that we need to save the average rewards and timesteps to failure
            ep_avg_reward = self.reward_sum / self._numSteps # only the average reward until failing
            name = f"tuning/rewards/mujoco_{self.alg_name}.npy"
            try:
                rewards = np.load(name)
            except FileNotFoundError:
                rewards = np.array([])

            rewards = np.append(rewards, ep_avg_reward)
            np.save(name, rewards)

            name_timesteps = f"tuning/timesteps/mujoco_{self.alg_name}.npy"
            try:
                timesteps = np.load(name_timesteps)
            except FileNotFoundError:
                timesteps = np.array([])
            timesteps = np.append(timesteps, self._numSteps)
            np.save(name_timesteps, timesteps)

            plot_rewards('mujoco', self.alg_name)
            
            # After saving, we want to reset episode-level statistics
            self._numSteps = 0
            self.reward_sum = 0
            self.landmark_history = []
        
        info = {}

        if self.render_mode == "human":
            self.render()
            
        print(reward, self.reward_sum, self.num_steps)
        return observation, reward, terminated, False, info