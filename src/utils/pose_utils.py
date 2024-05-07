import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytube import YouTube
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def compute_angle_3d(a_ind, b_ind, c_ind, points):
    """
    Compute the angle between three points.
    
    Args:
    a (tuple): The first point.
    b (tuple): The vertex point.
    c (tuple): The third point.
    
    Returns:
    float: The angle in degrees.
    """

    a = points[a_ind]
    b = points[b_ind]
    c = points[c_ind]

    a = [a.x, a.y, a.z]
    b = [b.x, b.y, b.z]
    c = [c.x, c.y, c.z]
    
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Create vectors ba and bc
    ab = a - b
    bc = c - b

    # Normalize the vectors
    ab = ab / np.linalg.norm(ab)
    bc = bc / np.linalg.norm(bc)
    
    # Convert the vectors to quaternions
    # The Rotation.from_rotvec function requires a rotation vector, which is a vector along the rotation axis with magnitude equal to the rotation angle.
    # We can create a rotation vector by multiplying the vector by the rotation angle, but in this case, we just want to convert the direction vectors to quaternions, so we can use the vectors directly.
    ab_quaternion = R.from_rotvec(ab).as_quat()
    bc_quaternion = R.from_rotvec(bc).as_quat()
    
    # Compute the quaternion difference
    # The Rotation.inv() function computes the inverse (conjugate) of a quaternion, and the Rotation.__mul__() function multiplies two quaternions.
    quaternion_difference = R.from_quat(ab_quaternion).inv() * R.from_quat(bc_quaternion)
    
    return quaternion_difference.as_quat()

def compute_angular_velocity(q1, q2, FRAME_DIFF, FRAMES_PER_SECOND):
    """
    Compute the angular velocity between two quaternions.
    
    Args:
    q1 (np.array): The first quaternion.
    q2 (np.array): The second quaternion.
    
    Returns:
    float: The angular velocity in radians per second.
    """

    dt = float(FRAME_DIFF)/FRAMES_PER_SECOND
    
    # Compute the quaternion difference
    q_diff = R.from_quat(q1).inv() * R.from_quat(q2)
    
    # Convert the quaternion difference to a rotation vector
    rotation_vector = R.from_quat(q_diff.as_quat()).as_rotvec()
    
    # Compute the rotation angle
    rotation_angle = np.linalg.norm(rotation_vector)
    
    # Compute the angular velocity
    angular_velocity = rotation_angle / dt
    
    return angular_velocity

def compute_velocity(p1, p2, FRAME_DIFF, FRAMES_PER_SECOND):
    """
    Compute the velocity at p2.
    
    Args:
    p1 (np.array): The position before p2.
    p2 (np.array): The position at which to compute the velocity.
    p3 (np.array): The position after p2.
    
    Returns:
    float: The velocity at p2 in meters per second.
    """
    dt = float(FRAME_DIFF)/FRAMES_PER_SECOND

    p1 = np.array(p1)
    p2 = np.array(p2)
   
    # Compute the velocities
    velocity = (p2 - p1) / dt
    return velocity

'''
Computes and extracts landmarks from frame (RGB image) using MediaPipe Pose.
'''
def extract_landmarks_from_frame(frame, mediapipe_pose):
    pos_dict = dict()
    angles_dict = dict()
    end_eff_pos_dict = dict()
    
    try:
        points = mediapipe_pose.process(frame).pose_landmarks.landmark
    except:
        print('could not extract points from this frame')
        return None
    
    pos_dict["origin"] = [points[0].x, points[0].y, points[0].z]
    pos_dict["right_shoulder"] = [points[12].x, points[12].y, points[12].z]
    pos_dict["left_shoulder"] = [points[11].x, points[11].y, points[11].z]
    pos_dict["right_elbow"] = [points[14].x, points[14].y, points[14].z]
    pos_dict["left_elbow"] = [points[13].x, points[13].y, points[13].z]
    pos_dict["right_hip"] = [points[24].x, points[24].y, points[24].z]
    pos_dict["left_hip"] = [points[23].x, points[23].y, points[23].z]
    pos_dict["right_knee"] = [points[26].x, points[26].y, points[26].z]
    pos_dict["left_knee"] = [points[25].x, points[25].y, points[25].z]
    pos_dict["right_ankle"] = [points[28].x, points[28].y, points[28].z]
    pos_dict["left_ankle"] = [points[27].x, points[27].y, points[27].z]
    pos_dict["right_wrist"] = [points[16].x, points[16].y, points[16].z]
    pos_dict["left_wrist"] = [points[15].x, points[15].y, points[15].z]
    
    end_eff_pos_dict["origin"] = [points[0].x, points[0].y, points[0].z]
    end_eff_pos_dict["right_ankle"] = [points[28].x, points[28].y, points[28].z]
    end_eff_pos_dict["left_ankle"] = [points[27].x, points[27].y, points[27].z]
    end_eff_pos_dict["right_wrist"] = [points[16].x, points[16].y, points[16].z]
    end_eff_pos_dict["left_wrist"] = [points[15].x, points[15].y, points[15].z]

    angles_dict["right_shoulder"] = compute_angle_3d(14, 12, 24, points)
    angles_dict["left_shoulder"] = compute_angle_3d(13, 11, 23, points)
    angles_dict["right_elbow"] = compute_angle_3d(16, 14, 12, points)
    angles_dict["left_elbow"] = compute_angle_3d(15, 13, 11, points)
    angles_dict["right_hip"] = compute_angle_3d(12, 24, 26, points)
    angles_dict["left_hip"] = compute_angle_3d(11, 23, 25, points)
    angles_dict["right_knee"] = compute_angle_3d(24, 26, 28, points)
    angles_dict["left_knee"] = compute_angle_3d(23, 25, 27, points)
    
    # Find the mean of all points that are values in pos_dict
    center_of_mass = np.mean([pos for pos in pos_dict.values()], axis=0)
    
    return {
        'pos': pos_dict, 
        'angle': angles_dict, 
        'end_eff_pos': end_eff_pos_dict,
        'center_of_mass': center_of_mass
    }

def compute_velocities_between_poses(prev_pos, curr_pos, FRAME_DIFF, FRAMES_PER_SECOND):
    if prev_pos is None:
        return None
    
    prev_pos_dict = prev_pos['pos']
    prev_angles_dict = prev_pos['angle']
    curr_pos_dict = curr_pos['pos']
    curr_angles_dict = curr_pos['angle']
    
    velocities_dict = dict()
    angle_velocities_dict = dict()
    
    for key in prev_pos_dict.keys():
        prev_pos = prev_pos_dict[key]
        curr_pos = curr_pos_dict[key]
        velocities_dict[key] = compute_velocity(prev_pos, curr_pos, FRAME_DIFF, FRAMES_PER_SECOND)
    
    for key in prev_angles_dict.keys():
        prev_angles = prev_angles_dict[key]
        curr_angles = curr_angles_dict[key]
        angle_velocities_dict[key] = compute_angular_velocity(prev_angles, curr_angles, FRAME_DIFF, FRAMES_PER_SECOND)
    
    return {
        'velocity': velocities_dict,
        'angle_velocity': angle_velocities_dict
    }
    
if __name__ == '__main__':
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Read an image from a file path
    image_path = '../../saved_frames/inputs/1.jpg'
    image = cv2.imread(image_path)

    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = extract_landmarks_from_frame(image_rgb, pose)
    print(results)