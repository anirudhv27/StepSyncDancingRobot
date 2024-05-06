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

def compute_angular_velocity_2(q1, q2, FRAME_DIFF, FRAMES_PER_SECOND):
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

def compute_angular_velocity(q1, q2, q3, FRAME_DIFF, FRAMES_PER_SECOND):
    """
    Compute the angular velocity at q2.
    
    Args:
    q1 (np.array): The quaternion before q2.
    q2 (np.array): The quaternion at which to compute the angular velocity.
    q3 (np.array): The quaternion after q2.
    
    Returns:
    float: The angular velocity at q2 in radians per second.
    """
    
    # Compute the angular velocities
    angular_velocity_12 = compute_angular_velocity_2(q1, q2, FRAME_DIFF, FRAMES_PER_SECOND)
    angular_velocity_23 = compute_angular_velocity_2(q2, q3, FRAME_DIFF, FRAMES_PER_SECOND)
    
    # Compute the average angular velocity
    angular_velocity = (angular_velocity_12 + angular_velocity_23) / 2
    
    return angular_velocity

def compute_velocity(p1, p2, p3, FRAME_DIFF, FRAMES_PER_SECOND):
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
    p3 = np.array(p3)
   
    # Compute the velocities
    velocity_12 = (p2 - p1) / dt
    velocity_23 = (p3 - p2) / dt
    
    # Compute the average velocity
    velocity = (velocity_12 + velocity_23) / 2
    
    return velocity

def compute_pos_angles(landmarks, FRAME_DIFF, FRAMES_PER_SECOND):
    pos = []
    angles = []

    for data_point in landmarks:
        pos_dict = dict()
        angles_dict = dict()
        angle_velocities_dict = dict()

        points = data_point.landmark
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

        angles_dict["right_shoulder"] = compute_angle_3d(14, 12, 24, points)
        angles_dict["left_shoulder"] = compute_angle_3d(13, 11, 23, points)
        angles_dict["right_elbow"] = compute_angle_3d(16, 14, 12, points)
        angles_dict["left_elbow"] = compute_angle_3d(15, 13, 11, points)
        angles_dict["right_hip"] = compute_angle_3d(12, 24, 26, points)
        angles_dict["left_hip"] = compute_angle_3d(11, 23, 25, points)
        angles_dict["right_knee"] = compute_angle_3d(24, 26, 28, points)
        angles_dict["left_knee"] = compute_angle_3d(23, 25, 27, points)

        pos.append(pos_dict)
        angles.append(angles_dict)

    angle_velocities = []
    velocities = []

    for i in range (FRAME_DIFF, len(angles) - FRAME_DIFF):
        angle_velocities_dict = dict()
        velocities_dict = dict()

        angles_dict = angles[i]
        pos_dict = pos[i]

        for key in angles_dict.keys():
            prev_angle = angles[i - 1][key]
            angle = angles_dict[key]
            next_angle = angles[i + 1][key]
            angle_velocities_dict[key] = compute_angular_velocity(prev_angle, angle, next_angle, FRAME_DIFF, FRAMES_PER_SECOND)       
        angle_velocities.append(angle_velocities_dict)

        for key in pos_dict.keys():
            prev_pos = pos[i - FRAME_DIFF][key]
            curr_pos = pos_dict[key]
            next_pos = pos[i + FRAME_DIFF][key]
            velocities_dict[key] = compute_velocity(prev_pos, curr_pos, next_pos, FRAME_DIFF, FRAMES_PER_SECOND)       
        angle_velocities.append(angle_velocities_dict)
        velocities.append(velocities_dict)
    
    pos = pos[FRAME_DIFF:len(pos) - FRAME_DIFF]
    angles = angles[FRAME_DIFF:len(angles) - FRAME_DIFF]

    return {
        'pos': pos, 
        'angles': angles, 
        'velocities': velocities, 
        'angle_velocities': angle_velocities
    }

