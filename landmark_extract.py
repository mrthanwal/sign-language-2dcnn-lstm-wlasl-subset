import os
import cv2
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Paths
frames_folder = "frames"
landmarks_folder = "landmarks"

# Ensure landmarks directory exists
os.makedirs(landmarks_folder, exist_ok=True)

# Define fixed shape for hand landmarks
NUM_KEYPOINTS = 21  # 21 keypoints per hand
NUM_HANDS = 2  # Max two hands per frame
FEATURES = NUM_KEYPOINTS * 3  # x, y, z for each keypoint
TOTAL_FEATURES = NUM_HANDS * FEATURES  # 2 hands * 21 keypoints * 3 features = 126

# Function to extract hand landmarks
def extract_hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        result = hands.process(image_rgb)  # Detect hands
        
        # Initialize zero-filled array (default: no hands detected)
        landmarks = np.zeros(TOTAL_FEATURES, dtype=np.float32)

        if result.multi_hand_landmarks:  # If hands detected
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if i >= 2:  # Only store up to 2 hands
                    break
                for j, lm in enumerate(hand_landmarks.landmark):
                    index = i * FEATURES + j * 3
                    landmarks[index:index+3] = [lm.x, lm.y, lm.z]

        return landmarks  # Returns a consistent 126-length array

# Function to process a subset of videos
def process_videos(video_list):
    for video_id in video_list:
        video_frames_path = os.path.join(frames_folder, video_id)
        if not os.path.isdir(video_frames_path):
            continue  # Skip if not a folder

        landmark_list = []  # Store landmarks for all frames in this video

        # Process each frame in sorted order
        for frame_filename in sorted(os.listdir(video_frames_path)):
            frame_path = os.path.join(video_frames_path, frame_filename)
            image = cv2.imread(frame_path)  # Read frame
            
            if image is not None:
                landmarks = extract_hand_landmarks(image)  # Extract landmarks
                landmark_list.append(landmarks)  # Append landmarks

        # Convert list to NumPy array (ensures uniform shape)
        landmark_array = np.array(landmark_list, dtype=np.float32)

        # Save extracted landmarks as .npz file
        if landmark_array.size > 0:
            np.savez(os.path.join(landmarks_folder, f"{video_id}.npz"), landmark_array)
            print(f"✅ Saved landmarks for {video_id}/")
        else:
            print(f"⚠️ Warning: No hand landmarks found for {video_id}, skipping.")

# Get list of video folders
all_videos = [v for v in os.listdir(frames_folder) if os.path.isdir(os.path.join(frames_folder, v))]
num_threads = 16  # Set the number of parallel threads

# Split video list into chunks for each thread
chunk_size = len(all_videos) // num_threads
video_chunks = [all_videos[i:i + chunk_size] for i in range(0, len(all_videos), chunk_size)]

# Run multithreading for parallel processing
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(process_videos, video_chunks)

print("🎯 Landmark extraction completed using multi-threading!")
