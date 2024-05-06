import gym
import pybullet_envs
from pybullet_envs.deep_mimic.gym_env.deep_mimic_env import HumanoidDeepBulletEnv
import matplotlib.pyplot as plt

import numpy as np

class CustomHumanoidDeepBulletEnv(HumanoidDeepBulletEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        
    def __init__(self, renders=False, arg_file='', test_mode=False,
                 time_step=1./240, rescale_actions=True, rescale_observations=True,
                 custom_cam_dist=4, custom_cam_pitch=0.1, custom_cam_yaw=45):
        
        super().__init__(renders=renders, arg_file=arg_file, test_mode=test_mode,
                         time_step=time_step, rescale_actions=rescale_actions, 
                         rescale_observations=rescale_observations)
        
        self._cam_dist = custom_cam_dist
        self._cam_pitch = custom_cam_pitch
        self._cam_yaw = custom_cam_yaw

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