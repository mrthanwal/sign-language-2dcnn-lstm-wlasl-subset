import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def detect_motion_start(cap, threshold=30, min_contour_area=500):
    """
    Detects the first frame where hand motion occurs.
    """
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Error: Could not read the first frame!")
        return 0  

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        frame_diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                print(f"Motion detected at frame {frame_idx}")
                return frame_idx  

        prev_frame = gray
        frame_idx += 1

    print("No motion detected. Starting from the beginning.")
    return 0  


def is_hand_visible(frame):
    """
    Detects if a hand is visible in the given frame using MediaPipe Hands.
    :param frame: The input frame (BGR format)
    :return: True if at least one hand is detected, otherwise False
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(frame_rgb)  

    return results.multi_hand_landmarks is not None  


def frame_capture(video_path, num_frames, output_folder):
    """
    Extracts frames from the video where hands are detected.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Error: Video {video_path} has 0 frames or failed to open.")
        return

    print(f"Total frames in video: {total_frames}")

    motion_start = detect_motion_start(cap)
    frame_indices = np.linspace(motion_start, total_frames - 1, num_frames * 2, dtype=int)  # Capture more frames initially

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = cap.read()
        
        if success and image is not None and is_hand_visible(image):  # Check if hand is visible
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, image)
            print(f"Saved frame {count} at index {frame_index} -> {frame_filename}")
            count += 1

            if count >= num_frames:  # Stop when required frames are collected
                break
        else:
            print(f"Skipping frame {frame_index} (No hand detected or read failure).")

    cap.release()


# Define input and output folders
folder = "final_dataset\\filtered_videos"
output_folder = "frames"

for video in os.listdir(folder):
    video_path = os.path.join(folder, video)

    # Extract video ID (without .mp4)
    video_id = os.path.splitext(video)[0]
    print(f"Extracting frames for video {video_id} from path {video_path}")

    video_out = os.path.join(output_folder, video_id)
    
    os.makedirs(video_out, exist_ok=True)

    frame_capture(video_path, 16, video_out)