%reset -f
#----------------------------------------------------
# Chapter One
#----------------------------------------------------
# 1.Pre-Processing (Body Motion Data)
# Parsing
# Data Interpolation/Resizing
# Denoising
# NaNs
# Normalization
# Standardization
#----------------------------------------------------

import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
import time

# Record the start time
start_time = time.time()

# Suppress warnings
warnings.filterwarnings("ignore")

# Parse BVH files
def parse_bvh(file_path):
    """Parse the BVH file to extract the header and motion data."""
    print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header, motion_data = [], []
    capture_data = False
    for line in lines:
        if "MOTION" in line:
            capture_data = True
        elif capture_data:
            if line.strip().startswith("Frames") or line.strip().startswith("Frame Time"):
                continue
            motion_data.append(np.fromstring(line, sep=' '))
        else:
            header.append(line)
    print(f"Finished parsing: {file_path}")
    return header, np.array(motion_data)

def read_bvh(filename):
    """Reads a BVH file and returns motion data parsed."""
    header, motion_data = parse_bvh(filename)
    return motion_data

def interpolate_frames(motion_data, target_frame_count):
    """Interpolates BVH frames to match a target frame count."""
    print(f"Interpolating frames to match target frame count: {target_frame_count}")
    original_frame_count = len(motion_data)
    original_time = np.linspace(0, 1, original_frame_count)
    target_time = np.linspace(0, 1, target_frame_count)
    interpolated_frames = []
    for frame in np.array(motion_data).T:
        interpolator = interp1d(original_time, frame.astype(float), kind='linear')
        interpolated_frame = interpolator(target_time)
        interpolated_frames.append(interpolated_frame)
    print(f"Finished interpolation for frames: Original frame count = {original_frame_count}, Target frame count = {target_frame_count}")
    return np.array(interpolated_frames).T

def find_max_frames(folder_path):
    """Finds the maximum number of frames among all BVH files in a folder."""
    print(f"Finding maximum number of frames in folder: {folder_path}")
    max_frames = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            motion_data = read_bvh(os.path.join(folder_path, filename))
            max_frames = max(max_frames, len(motion_data))
    print(f"Maximum frames found: {max_frames}")
    return max_frames

def process_bvh_files(folder_path, max_frames):
    """Processes each BVH file in the folder after interpolating to the same number of frames."""
    print(f"Processing BVH files in folder: {folder_path}")
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            print(f"Processing file: {filename}")
            full_path = os.path.join(folder_path, filename)
            motion_data = read_bvh(full_path)
            interpolated_frames = interpolate_frames(motion_data, max_frames)
            all_features.append(interpolated_frames.flatten())  # Flatten the data for processing
            print(f"Finished processing file: {filename}")
    print(f"Finished processing all BVH files in folder: {folder_path}")
    return np.array(all_features)

# -------------------------------
# Train Folder
train_folder_path = 'Small Dataset/'
# -------------------------------

# Find maximum frame size in the training data
print("Finding maximum frames in the training dataset...")
max_frames_train = find_max_frames(train_folder_path)

# Process BVH files using the maximum frame size
print("Processing training data...")
all_features_train = process_bvh_files(train_folder_path, max_frames_train)

# Train Labels
print("Assigning labels to the dataset...")
C_Angry = [0] * 42
C_Depressed = [1] * 42
C_Neutral = [2] * 42
C_Proud = [3] * 32
Labels = np.array(C_Angry + C_Depressed + C_Neutral + C_Proud, dtype=np.int32)
print("Labels assigned.")

# Standardize the data
print("Standardizing the data...")
scaler = StandardScaler()
X_standardized = scaler.fit_transform(all_features_train)
print("Data standardized.")

# Normalize the data (min-max scaling)
print("Normalizing the data...")
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standardized)
print("Data normalized.")

# Handle NaN values
print("Handling NaN values...")
imputer = SimpleImputer(strategy='mean')
X_cleaned = imputer.fit_transform(X_normalized)
print("NaN values handled.")

# Final feature set and labels
X = X_cleaned
Y = Labels

# Print completion message
end_time = time.time()
print(f"Pre-processing completed in {end_time - start_time:.2f} seconds.")
