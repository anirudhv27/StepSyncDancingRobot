import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_diff_sum(quat1, quat2):
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
    
def calc_reward(target_poses, numSteps, agent_id, curr_landmarks, ep_done):
    # get pose of the current position
    if ep_done or curr_landmarks is None: # we are about to reset the environment: what reward to give?
        return 0 # dont want to put in these type of positions!
    
    # get target pose
    # print(numSteps)
    target_pose = target_poses[numSteps]

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
    angle_quat_diff = np.exp(-2 * quat_diff_sum(agent_angle, target_angle))
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

    return reward