import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Load trained .h5 Keras model
model = tf.keras.models.load_model("99_accurate_model.h5")

# Load label mappings
label_map = {
    0: 'apple', 1: 'appointment', 2: 'argue', 3: 'bad', 4: 'balance', 5: 'bar', 6: 'because',
    7: 'bird', 8: 'black', 9: 'blanket', 10: 'brother', 11: 'champion', 12: 'cheat',
    13: 'check', 14: 'convince', 15: 'cry', 16: 'daughter', 17: 'deaf', 18: 'delay',
    19: 'delicious', 20: 'doctor', 21: 'dog', 22: 'environment', 23: 'example',
    24: 'family', 25: 'far', 26: 'fat', 27: 'fish', 28: 'full', 29: 'give',
    30: 'good', 31: 'government', 32: 'graduate', 33: 'hot', 34: 'interest',
    35: 'language', 36: 'laugh', 37: 'leave', 38: 'letter', 39: 'like',
    40: 'many', 41: 'mother', 42: 'move', 43: 'no', 44: 'orange', 45: 'order',
    46: 'perspective', 47: 'play', 48: 'ready', 49: 'room', 50: 'sandwich',
    51: 'score', 52: 'secretary', 53: 'silly', 54: 'snow', 55: 'son', 56: 'soon',
    57: 'speech', 58: 'study', 59: 'sweet', 60: 'take', 61: 'tell', 62: 'theory',
    63: 'thursday', 64: 'toast', 65: 'wait', 66: 'walk', 67: 'white',
    68: 'why', 69: 'woman', 70: 'work', 71: 'write', 72: 'year', 73: 'yesterday'
}

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture(0)
WINDOW_SIZE = 15  # Sliding window size
frame_window = deque(maxlen=WINDOW_SIZE)
landmark_buffer = deque(maxlen=WINDOW_SIZE)

# Lock for thread safety
lock = threading.Lock()

# Store last frame with landmarks
last_landmark_frame = None

def extract_landmarks():
    """
    Function to extract hand landmarks in a separate thread.
    """
    global last_landmark_frame
    while True:
        if len(frame_window) == WINDOW_SIZE:
            keypoints_list = []
            with lock:
                frames = list(frame_window)  # Copy frames

            # Process only the last frame for landmark visualization
            last_frame = frames[-1]
            image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            keypoints = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])

                    # Draw landmarks on the last frame
                    last_landmark_frame = last_frame.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(last_landmark_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(keypoints) == 126:  # Ensure both hands detected
                keypoints_list.append(keypoints)

            with lock:
                if keypoints_list:
                    landmark_buffer.append(keypoints_list[-1])  # Store last valid landmarks

# Start landmark extraction in a separate thread
landmark_thread = threading.Thread(target=extract_landmarks, daemon=True)
landmark_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        frame_window.append(frame.copy())  # Store frames for processing

    # Show the frame with landmarks
    with lock:
        display_frame = last_landmark_frame if last_landmark_frame is not None else frame.copy()

    if len(landmark_buffer) == WINDOW_SIZE:
        with lock:
            input_data = np.array(landmark_buffer).reshape(1, WINDOW_SIZE, 126, 1).astype(np.float32)

        # Run inference using .h5 Keras model
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display prediction
        if confidence > 0.7:
            label = label_map.get(predicted_class, "Unknown")
            cv2.putText(display_frame, f"Sign: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Sign Language Recognition', display_frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
