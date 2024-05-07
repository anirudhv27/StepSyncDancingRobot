import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytube import YouTube
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from compute_pos_angles import compute_pos_angles
import matplotlib.pyplot as plt
FRAMES_PER_SECOND = 30
FRAME_DIFF = 3

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

# Process each video frame.
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    writer = setup_video_writer(cap, output_video_path)
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
        
        # Make detection.
        results = pose.process(image)

        if (results.pose_landmarks == None):
            continue
        landmarks.append(results.pose_landmarks)
        
        # Draw the pose annotations on the image.
        keypoints = np.zeros((height, width, channels), dtype="uint8")
        
        mp_pose.POSE_CONNECTIONS
        mp_drawing = mp.solutions.drawing_utils
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(keypoints, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the frame with annotations.
        writer.write(keypoints)

    cap.release()
    writer.release()
    return landmarks
    
def gen_dataset_from_url(danceURL, filename='dance'):
    input_path = f'../data/{filename}.mp4'
    output_path = f'../keypoints/{filename}.mp4'
    download_youtube_video(danceURL, input_path)
    landmarks = process_video(f"../data/{filename}.mp4", output_path)
    print ("Gen dataset")
    print (landmarks)
    dataset = compute_pos_angles(landmarks, FRAME_DIFF, FRAMES_PER_SECOND)
    return dataset

gen_dataset_from_url("https://www.youtube.com/watch?v=9TWj9I3CKzg&pp=ygUaYmF0YW1peCBkaWwgZGFuY2UgdHV0b3JpYWw%3D")
if __name__ == '__main__':
    danceURL = 'https://www.youtube.com/watch?v=9TWj9I3CKzg'
    filename = 'dance'
    input_path = f'../../data/{filename}.mp4'
    output_path = f'../../keypoints/{filename}.mp4'
    download_youtube_video(danceURL, input_path)
    # Example usage:
    landmarks = process_video(input_path, output_path)