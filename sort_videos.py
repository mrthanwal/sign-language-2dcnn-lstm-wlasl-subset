import os
import json
import shutil

# Paths
json_file = "dataset\\WLASL_v0.3_cleaned.json"  # JSON file with sign language data
video_folder = "selected_videos"  # Folder containing all downloaded videos
output_folder = "sorted_videos"  # Output folder for class-based organization

# Load JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Create a mapping of video_id to gloss (sign word)
video_id_to_gloss = {}
for sign_entry in data:
    gloss = sign_entry["gloss"]  # Extract sign word
    for instance in sign_entry["instances"]:
        video_id = instance["video_id"]  # Extract video ID
        video_id_to_gloss[video_id] = gloss  # Map video ID to class

# Process each video in selected_videos folder
for video_filename in os.listdir(video_folder):
    if not video_filename.endswith(".mp4"):
        continue  # Skip non-video files

    video_id = os.path.splitext(video_filename)[0]  # Remove .mp4 extension

    if video_id in video_id_to_gloss:  # Check if video ID exists in JSON
        gloss = video_id_to_gloss[video_id]  # Get corresponding sign word
        class_folder = os.path.join(output_folder, gloss)  # Folder for class

        # Create the class folder if it doesn't exist
        os.makedirs(class_folder, exist_ok=True)

        # Move the video to its corresponding class folder
        video_path = os.path.join(video_folder, video_filename)
        shutil.move(video_path, os.path.join(class_folder, video_filename))
        print(f"Moved {video_filename} -> {class_folder}/")
    else:
        print(f"Warning: Video {video_filename} not found in JSON file.")

print("Selected videos sorted successfully!")
