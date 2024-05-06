import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytube import YouTube
import numpy as np

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
        # print(image.shape)
        image.flags.writeable = False
        height, width, channels = image.shape
        
        # Make detection.
        results = pose.process(image)
        landmarks.append(results.pose_landmarks)

        # if (results.pose_landmarks == None):
            # print (len(landmarks))
        
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

    # print(len(landmarks))
    # print(landmarks[1000])
    # print(type(landmarks))
    return landmarks

if __name__ == '__main__':
    # for i in range(9):
    #     input_image_path = f'../saved_frames/inputs/{i}.jpg'
    #     output_image_path = f'../saved_frames/outputs/{i}.jpg'
    #     # Initialize MediaPipe Pose.
    #     print('initializing mp_pose')
    #     mp_pose = mp.solutions.pose
    #     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) # , model_complexity={0,1,2} (fastest to slowest)
        
    #     # Read the image from input_image_path and use pose.process to extract landmarks.
    #     # Then, write the landmarks to a new image using mp.solutions.drawing_utils.draw_landmarks.
    #     print('reading image')
    #     image = cv2.imread(input_image_path)
    #     print(image.shape)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image.flags.writeable = False
        
    #     height, width, channels = image.shape
    #     results = pose.process(image)
    #     keypoints = np.zeros((height, width, channels), dtype="uint8")
        
    #     mp_drawing = mp.solutions.drawing_utils
    #     if results.pose_landmarks:
    #         mp_drawing.draw_landmarks(keypoints, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
    #     cv2.imwrite(output_image_path, keypoints)
    # danceURL = input('Link your desired dance here: ')
    
    danceURL = 'https://www.youtube.com/watch?v=9TWj9I3CKzg'
    # filename = input('What are you labeling this file? ')
    filename = 'badtameez_dil'
    input_path = f'../data/{filename}.mp4'
    output_path = f'../keypoints/{filename}.mp4'
    download_youtube_video(danceURL, input_path)
    # Example usage:
    landmarks = process_video(input_path, output_path)