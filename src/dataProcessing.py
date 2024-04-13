import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytube import YouTube

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

def process_video(video_path):
    # Initialize mediapipe pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # Capture video
    cap = cv2.VideoCapture(video_path)
    plt.figure(figsize=(12, 6))
    ims = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image to get the pose
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        mp_draw = mp.solutions.drawing_utils
        annotated_image = rgb_frame.copy()
        mp_draw.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert RGB to BGR for OpenCV
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Display side by side
        im = plt.imshow(cv2.hconcat([frame, annotated_image]), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(plt.gcf(), ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()

    cap.release()

download_youtube_video('https://www.youtube.com/watch?v=9TWj9I3CKzg', '../data/bollywood.mp4')
# Example usage:
process_video("../data/bollywood.mp4")
