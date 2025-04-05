import os
import json
import shutil

# Load class names (glosses) from class_list.txt
with open("class_list.txt", "r") as f:
    class_list = [line.strip().strip("',") for line in f if line.strip()]

# Load JSON file (update the path to your JSON file)
with open("dataset\\WLASL_v0.3_cleaned.json", "r") as f:
    data = json.load(f)

# Create destination folder
os.makedirs("selected_videos", exist_ok=True)

# Set of video_ids to avoid duplicates
moved_video_ids = set()

# Loop through JSON entries
for entry in data:
    gloss = entry.get("gloss", "").strip()
    if gloss in class_list:
        for instance in entry["instances"]:
            video_id = instance["video_id"]
            video_filename = f"{video_id}.mp4"
            src = os.path.join("videos", video_filename)
            dst = os.path.join("selected_videos", video_filename)

            if video_id not in moved_video_ids:
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"Moved: {video_filename}")
                    moved_video_ids.add(video_id)
                else:
                    print(f"Missing file: {video_filename}")
