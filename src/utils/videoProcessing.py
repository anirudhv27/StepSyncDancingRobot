import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytube import YouTube
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from pose_utils import compute_velocities_between_poses, extract_landmarks_from_frame
import matplotlib.pyplot as plt

def download_youtube_video(url, path):
    """
    Downloads a YouTube video from the given URL and saves it as an MP4 file.
    
    Args:
    url (str): The URL of the YouTube video.
    path (str): The path where the video will be saved, including the filename.
    
    Returns:
    bool: True if the download was successful, False otherwise.
    """
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Select the highest resolution stream available
        stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        
        # Download the video
        stream.download(filename=path)
        
        print(f"Video downloaded successfully: {path}")
        return True
    except Exception as e:
        print(f"Failed to download video: {str(e)}")
        return False

# Initialize the video writer.
def setup_video_writer(cap, output_filename='output_video.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_filename, fourcc, fps, (width, height))


'''
Processes video into a list of dictionary landmarks using MediaPipe Pose.
If landmarks[i] is None, then the frame was not processed successfully.

'''
def process_video_to_landmarks(input_video_path, FRAME_DIFF, FRAMES_PER_SECOND):
    cap = cv2.VideoCapture(input_video_path)
    # Initialize MediaPipe Pose.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) # , model_complexity={0,1,2} (fastest to slowest)

    landmarks = []
    print('Processing video...')
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        height, width, channels = image.shape
        
        curr_landmarks = extract_landmarks_from_frame(image, pose)
        if (curr_landmarks is None): # Handling failures by 
            landmarks.append(None)
            continue
        
        velocity_landmarks = None
        if len(landmarks) >= FRAME_DIFF:
            prev_landmarks = landmarks[-FRAME_DIFF]
            velocity_landmarks = compute_velocities_between_poses(prev_landmarks, curr_landmarks, FRAME_DIFF, FRAMES_PER_SECOND)
            
            # velocity_landmarks is None when prev_landmarks is None
            # Handle by setting velocity_landmarks to the velocity of the previous frame (best approximation)
            if velocity_landmarks is not None:
                curr_landmarks['velocity'] = velocity_landmarks['velocity']
                curr_landmarks['angle_velocity'] = velocity_landmarks['angle_velocity']
        
        landmarks.append(curr_landmarks)
        
    return landmarks
    
def generate_dataset_from_url(danceURL, filename='dance', FRAME_DIFF=3, FRAMES_PER_SECOND=30):
    input_path = f'../data/{filename}.mp4'
    download_youtube_video(danceURL, input_path)
    dataset = process_video_to_landmarks(f"../data/{filename}.mp4", FRAME_DIFF, FRAMES_PER_SECOND)
    return dataset