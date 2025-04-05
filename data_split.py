import os
import json
import numpy as np

# Paths
landmarks_folder = "landmarks"  # Source of landmarks
split_folder = "final_data_split"  # Destination for split data
json_file = "dataset\\WLASL_v0.3_cleaned.json"  # Your JSON file with split info

# Ensure split_data structure exists
os.makedirs(os.path.join(split_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(split_folder, "val"), exist_ok=True)
os.makedirs(os.path.join(split_folder, "test"), exist_ok=True)

# Load JSON annotations
with open(json_file, "r") as f:
    annotations = json.load(f)

# Create a mapping of video_id to split type
video_splits = {}  # {video_id: "train" / "val" / "test"}
classes = {}

for entry in annotations:
    gloss = entry["gloss"]
    for instance in entry["instances"]:
        video_splits[instance["video_id"]] = instance["split"]  # Store split info
        classes[instance["video_id"]] = gloss

# Store data for X (features) and Y (labels)
split_data = {
    "train": {"X": [], "Y": []},
    "val": {"X": [], "Y": []},
    "test": {"X": [], "Y": []},
}

# Process each class folder
#for class_name in os.listdir(landmarks_folder):
#    class_path = os.path.join(landmarks_folder, class_name)
#    if not os.path.isdir(class_path):
#        continue  # Skip non-folder items

    # Process each video file in the class folder
for video_file in os.listdir(landmarks_folder):
    video_id = os.path.splitext(video_file)[0]  # Remove ".npz"
    video_path = os.path.join(landmarks_folder, video_file)

    # Find the split category
    split = video_splits.get(video_id, "train")  # Default to "train" if not found
    split_folder_path = os.path.join(split_folder, split)

    # Load the landmarks
    data = np.load(video_path)["arr_0"]  # Load the landmark array

    # Append to split dataset
    split_data[split]["X"].append(data)
    split_data[split]["Y"].append(classes.get(video_id))  # Label is the class name

# Save split datasets
for split in ["train", "val", "test"]:
    x_save_path = os.path.join(split_folder, f"x_{split}.npz")
    y_save_path = os.path.join(split_folder, f"y_{split}.npz")

    np.savez(x_save_path, *split_data[split]["X"])  # Save X (features)
    np.savez(y_save_path, *split_data[split]["Y"])  # Save Y (labels)

    print(f"✅ Saved {split} data: {x_save_path}, {y_save_path}")

print("🎯 Data splitting completed!")
